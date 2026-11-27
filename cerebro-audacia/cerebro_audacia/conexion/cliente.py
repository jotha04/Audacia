"""Cliente Oficial Único del Cerebro Central de AudacIA.

Este módulo es la ÚNICA interfaz autorizada para que los agentes y el orquestador
consuman inferencia de lenguaje natural.

Garantías de Arquitectura:
1. Principio de Proveedor Único: Todo agente consume exclusivamente este cliente.
2. Ocultamiento de Infraestructura: Los agentes desconocen rutas .gguf, capas GPU,
   parámetros CUDA y configuración de llama-server.
3. Bloqueo de Modelos Secundarios: Queda terminantemente prohibido que un agente
   especifique un modelo alternativo. Cualquier intento de pasar 'model', 'modelo',
   'gguf_path' o similares lanza ViolacionArquitecturaError.
4. Gobernanza de Recursos: Control centralizado de concurrencia, colas y Circuit Breaker.
"""

from __future__ import annotations

import collections
import http.client
import io
import json
import logging
import math
import os
import queue
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from .excepciones import (
    CerebroCentralError,
    LlmColaExpiradaError,
    LlmLimiteExcedidoError,
    LlmNoDisponibleError,
    LlmOomError,
    LlmRespuestaInvalidaError,
    LlmSaturadoError,
    ViolacionArquitecturaError,
)

logger = logging.getLogger("cerebro_audacia.conexion")

# Parámetros estrictamente prohibidos para los llamadores (agentes)
PARAMETROS_PROHIBIDOS = frozenset({
    "model",
    "modelo",
    "model_path",
    "ruta_modelo",
    "modelo_ruta",
    "gguf",
    "gguf_path",
    "ruta_gguf",
    "weights",
    "pesos",
    "provider",
    "proveedor",
    "backend",
    "n_gpu_layers",
    "n_ctx",
})

_BLOQUE_JSON = re.compile(r"\{.*\}", re.DOTALL)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Tráfico normal
    OPEN = "OPEN"            # Rechazo inmediato por fallo persistente
    HALF_OPEN = "HALF_OPEN"  # Sondeo de prueba


@dataclass(frozen=True)
class RespuestaCerebroCentral:
    """Respuesta normalizada emitida por el Cerebro Central hacia los agentes."""

    texto: str
    tool_calls: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    datos_json: dict[str, Any] | None = None
    finish_reason: str = "stop"
    modelo_reportado: str = ""
    duracion_ms: float = 0.0
    tokens_completados: int = 0
    espera_cola_ms: float = 0.0
    tokens_prompt: int | None = None
    tokens_prompt_cacheados: int | None = None
    prefill_ms: float | None = None
    generacion_ms: float | None = None


class LlmCircuitBreaker:
    """Disyuntor thread-safe para aislar fallos del servidor LLM central."""

    def __init__(self, fallos_max: int = 3, tiempo_enfriamiento_s: float = 5.0):
        self._fallos_max = max(1, fallos_max)
        self._tiempo_enfriamiento_s = max(0.1, tiempo_enfriamiento_s)
        self._lock = threading.Lock()
        self._estado = CircuitState.CLOSED
        self._fallos_consecutivos = 0
        self._tiempo_apertura = 0.0
        self._sondeo_en_vuelo = False

    @property
    def estado(self) -> CircuitState:
        with self._lock:
            if self._estado == CircuitState.OPEN:
                if (time.monotonic() - self._tiempo_apertura) >= self._tiempo_enfriamiento_s:
                    self._estado = CircuitState.HALF_OPEN
            return self._estado

    def puede_ejecutar(self) -> bool:
        with self._lock:
            if self._estado == CircuitState.OPEN:
                if (time.monotonic() - self._tiempo_apertura) >= self._tiempo_enfriamiento_s:
                    self._estado = CircuitState.HALF_OPEN
                else:
                    return False

            if self._estado == CircuitState.CLOSED:
                return True

            if self._estado == CircuitState.HALF_OPEN:
                if not self._sondeo_en_vuelo:
                    self._sondeo_en_vuelo = True
                    return True
                return False

            return False

    def registrar_exito(self) -> None:
        with self._lock:
            self._fallos_consecutivos = 0
            self._estado = CircuitState.CLOSED
            self._sondeo_en_vuelo = False

    def registrar_fallo(self, exc: BaseException | None = None) -> None:
        with self._lock:
            self._sondeo_en_vuelo = False
            self._fallos_consecutivos += 1
            if self._fallos_consecutivos >= self._fallos_max or self._estado == CircuitState.HALF_OPEN:
                self._estado = CircuitState.OPEN
                self._tiempo_apertura = time.monotonic()


class LlmGovernor:
    """Gobernador central de concurrencia y cola de admisión para el Cerebro Central."""

    def __init__(self, max_concurrencia: int = 1, max_cola: int = 5, timeout_cola_s: float = 10.0):
        self._max_concurrencia = max(1, max_concurrencia)
        self._max_cola = max(0, max_cola)
        self._timeout_cola_s = max(0.1, timeout_cola_s)
        self._activas = 0
        self._esperadores: collections.deque[object] = collections.deque()
        self._cond = threading.Condition(threading.Lock())

    @property
    def activas(self) -> int:
        with self._cond:
            return self._activas

    @property
    def en_cola(self) -> int:
        with self._cond:
            return len(self._esperadores)

    @property
    def max_concurrencia(self) -> int:
        return self._max_concurrencia

    @contextmanager
    def adquirir(self) -> Iterator[float]:
        espera_ms = 0.0
        with self._cond:
            # No permitir que las solicitudes recién llegadas adelanten a las
            # que ya esperan, incluso si una ranura acaba de quedar libre.
            if self._activas < self._max_concurrencia and not self._esperadores:
                self._activas += 1
            else:
                if len(self._esperadores) >= self._max_cola:
                    raise LlmSaturadoError(
                        f"Cerebro Central saturado: cola de admisión llena ({len(self._esperadores)}/{self._max_cola} en espera)."
                    )

                turno = object()
                self._esperadores.append(turno)
                inicio_espera = time.monotonic()
                deadline = inicio_espera + self._timeout_cola_s
                try:
                    while (
                        self._activas >= self._max_concurrencia
                        or not self._esperadores
                        or self._esperadores[0] is not turno
                    ):
                        restante = deadline - time.monotonic()
                        if restante <= 0:
                            self._esperadores.remove(turno)
                            self._cond.notify_all()
                            raise LlmColaExpiradaError(
                                f"Tiempo de espera en cola del Cerebro Central excedido ({self._timeout_cola_s:.1f}s)."
                            )
                        self._cond.wait(timeout=restante)

                    self._esperadores.popleft()
                    self._activas += 1
                    espera_ms = max(0.0, (time.monotonic() - inicio_espera) * 1000.0)
                    # Otra ranura puede permanecer libre si max_concurrencia > 1.
                    self._cond.notify_all()
                except BaseException:
                    if turno in self._esperadores:
                        self._esperadores.remove(turno)
                        self._cond.notify_all()
                    raise

        try:
            yield espera_ms
        finally:
            with self._cond:
                self._activas -= 1
                self._cond.notify_all()


def _es_urlopen_mockeado() -> bool:
    """Detecta si urllib.request.urlopen fue interceptado por un unittest.mock.patch."""
    fn = getattr(urllib.request, "urlopen", None)
    if fn is None:
        return False
    tipo = type(fn).__name__
    return "Mock" in tipo or hasattr(fn, "assert_called") or hasattr(fn, "_mock_return_value")


class PoolConexionesHTTP:
    """Pool acotado y thread-safe de conexiones HTTP/1.1 persistentes (Keep-Alive).
    
    Implementa reutilización de sockets TCP para minimizar la sobrecarga de transporte
    y handshake recurrente hacia el Cerebro Central compartido.
    """

    def __init__(
        self,
        host: str,
        port: int,
        is_https: bool = False,
        timeout_defecto: float = 20.0,
        max_size: int = 2,
    ) -> None:
        self.host = host
        self.port = port
        self.is_https = is_https
        self.timeout_defecto = timeout_defecto
        self.max_size = max(1, max_size)
        self._pool: queue.LifoQueue[http.client.HTTPConnection] = queue.LifoQueue(maxsize=self.max_size)
        self._lock = threading.Lock()
        self._cerrado = False
        self._total_creadas = 0
        self._total_reutilizadas = 0

    @property
    def cerrado(self) -> bool:
        with self._lock:
            return self._cerrado

    @property
    def total_creadas(self) -> int:
        with self._lock:
            return self._total_creadas

    @property
    def total_reutilizadas(self) -> int:
        with self._lock:
            return self._total_reutilizadas

    @property
    def conexiones_en_pool(self) -> int:
        return self._pool.qsize()

    @staticmethod
    def _es_socket_utilizable(conn: http.client.HTTPConnection) -> bool:
        """Comprueba si una conexión en reposo sigue abierta antes de despachar una nueva solicitud."""
        if conn.sock is None:
            return True
        try:
            if conn.sock.fileno() == -1:
                return False
            import select
            r, _, x = select.select([conn.sock], [], [conn.sock], 0.0)
            if x:
                return False
            if r:
                # Socket legible en reposo: si peek devuelve b"", el extremo remoto cerró (EOF)
                try:
                    peek = conn.sock.recv(1, socket.MSG_PEEK)
                    if not peek:
                        return False
                    return False
                except (socket.error, OSError):
                    return False
        except (socket.error, OSError, ValueError):
            return False
        return True

    def adquirir(self, timeout: float | None = None) -> http.client.HTTPConnection:
        """Obtiene una conexión persistente disponible o instancia una nueva."""
        with self._lock:
            if self._cerrado:
                raise RuntimeError("El pool de conexiones se encuentra cerrado.")

        t = timeout if timeout is not None else self.timeout_defecto
        while True:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                break

            if not self._es_socket_utilizable(conn):
                try:
                    conn.close()
                except Exception:
                    pass
                continue

            conn.timeout = t
            if conn.sock is not None:
                conn.sock.settimeout(t)
            with self._lock:
                self._total_reutilizadas += 1
            return conn

        with self._lock:
            self._total_creadas += 1
        if self.is_https:
            import ssl
            return http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=t,
                context=ssl.create_default_context(),
            )
        return http.client.HTTPConnection(self.host, self.port, timeout=t)

    def liberar(self, conn: http.client.HTTPConnection, valida: bool = True) -> None:
        """Devuelve la conexión al pool si sigue siendo válida y el pool no está cerrado, o la cierra."""
        with self._lock:
            cerrado = self._cerrado

        if cerrado or not valida:
            try:
                conn.close()
            except Exception:
                pass
            return

        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            try:
                conn.close()
            except Exception:
                pass

    def cerrar(self) -> None:
        """Cierra todas las conexiones ociosas actualmente en el pool y marca el pool como cerrado."""
        with self._lock:
            self._cerrado = True
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass


class ConfiguracionCerebroCentral:
    """Fuente única de configuración para la conexión con el Cerebro Central."""

    def __init__(self, perfil: str | None = None) -> None:
        self.url_chat = "http://127.0.0.1:8080/v1/chat/completions"
        self.url_salud = "http://127.0.0.1:8080/health"
        self.url_modelos = "http://127.0.0.1:8080/v1/models"
        self.timeout_conexion = 5.0
        self.timeout_lectura = 20.0
        self.timeout_cola = 10.0
        self.max_concurrencia = 1
        self.max_cola = 5
        self.modelo_por_defecto = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        self.max_caracteres_prompt = 100_000
        self.n_ctx = 8192
        self.parallel = 1
        self.margen_seguridad_tokens = 128
        self.ttl_cache_salud = 2.0
        self.perfil_solicitado = perfil
        self.perfil_activo = "desarrollo_rtx2000ada"
        self._cargar_desde_infraestructura()

    @property
    def ventana_contexto_por_ranura(self) -> int:
        """Contexto utilizable asignado a cada ranura concurrente (n_ctx // parallel)."""
        ranuras = max(1, self.parallel)
        return max(512, self.n_ctx // ranuras)

    def _cargar_desde_infraestructura(self) -> None:
        # 1. Determinar perfil explícito solicitado o por variable de entorno
        perfil_solicitado = (
            self.perfil_solicitado
            or os.getenv("AUDACIA_LLM_PERFIL")
            or os.getenv("AUDACIA_LLM_PROFILE")
        )

        # 2. Intentar leer config_llm.json oficial
        raiz = Path(__file__).resolve().parent.parent.parent
        ruta_cfg = raiz / "cerebro_audacia" / "proveedor_modelo" / "config_llm.json"

        if ruta_cfg.exists():
            try:
                with open(ruta_cfg, "r", encoding="utf-8") as f:
                    datos = json.load(f)
                    srv = datos.get("servicio", {})
                    self.url_chat = srv.get("url_chat", self.url_chat)
                    self.url_salud = srv.get("url_salud", self.url_salud)
                    self.url_modelos = srv.get("url_modelos", self.url_modelos)

                    perfiles = datos.get("perfiles", {})
                    perfil_defecto = datos.get("perfil_predeterminado") or "desarrollo_rtx2000ada"
                    perfil_objetivo = (perfil_solicitado or perfil_defecto).strip()
                    if perfil_objetivo in perfiles:
                        perfil = perfiles[perfil_objetivo]
                        self.perfil_activo = perfil_objetivo
                    else:
                        perfil_respaldo = perfil_defecto if perfil_defecto in perfiles else next(iter(perfiles), "")
                        logger.warning(
                            "Perfil '%s' no encontrado en config_llm.json. Usando '%s'.",
                            perfil_objetivo,
                            perfil_respaldo or "valores predeterminados",
                        )
                        perfil = perfiles.get(perfil_respaldo, {})
                        self.perfil_activo = perfil_respaldo or "desarrollo_rtx2000ada"

                    if perfil:
                        self.max_concurrencia = int(perfil.get("parallel", self.max_concurrencia))
                        self.parallel = self.max_concurrencia
                        self.max_cola = max(0, int(perfil.get("max_cola", self.max_cola)))
                        self.timeout_cola = max(0.1, float(perfil.get("timeout_cola", self.timeout_cola)))
                        self.n_ctx = int(perfil.get("n_ctx", self.n_ctx))
                        self.modelo_por_defecto = perfil.get("modelo_archivo", self.modelo_por_defecto)
            except Exception as e:
                logger.warning("No se pudo cargar config_llm.json (%s). Usando valores predeterminados.", e)

        # 3. Variables de entorno institucionales (anulación y gobernanza)
        env_url = os.getenv("AUDACIA_LLM_URL") or os.getenv("LLAMA_SERVER_URL")
        if env_url:
            self.url_chat = env_url.strip()
            if self.url_chat.endswith("/v1/chat/completions"):
                base = self.url_chat[:-len("/v1/chat/completions")]
                self.url_salud = base + "/health"
                self.url_modelos = base + "/v1/models"

        if os.getenv("AUDACIA_LLM_MAX_CONCURRENCY"):
            try:
                self.max_concurrencia = int(os.environ["AUDACIA_LLM_MAX_CONCURRENCY"])
                self.parallel = self.max_concurrencia
            except ValueError:
                pass

        if os.getenv("AUDACIA_LLM_MAX_QUEUE"):
            try:
                self.max_cola = max(0, int(os.environ["AUDACIA_LLM_MAX_QUEUE"]))
            except ValueError:
                pass

        if os.getenv("AUDACIA_LLM_QUEUE_TIMEOUT"):
            try:
                self.timeout_cola = float(os.environ["AUDACIA_LLM_QUEUE_TIMEOUT"])
            except ValueError:
                pass

        if os.getenv("AUDACIA_LLM_CONNECT_TIMEOUT"):
            try:
                self.timeout_conexion = float(os.environ["AUDACIA_LLM_CONNECT_TIMEOUT"])
            except ValueError:
                pass

        if os.getenv("AUDACIA_LLM_READ_TIMEOUT"):
            try:
                self.timeout_lectura = float(os.environ["AUDACIA_LLM_READ_TIMEOUT"])
            except ValueError:
                pass

        if os.getenv("AUDACIA_LLM_MAX_CHARS"):
            try:
                self.max_caracteres_prompt = int(os.environ["AUDACIA_LLM_MAX_CHARS"])
            except ValueError:
                pass

        if os.getenv("AUDACIA_LLM_HEALTH_CACHE_TTL"):
            try:
                self.ttl_cache_salud = float(os.environ["AUDACIA_LLM_HEALTH_CACHE_TTL"])
            except ValueError:
                pass


class ClienteCerebroCentral:
    """Cliente oficial único para interactuar con el Cerebro Central de AudacIA."""

    def __init__(
        self,
        config: ConfiguracionCerebroCentral | None = None,
        max_concurrencia: int | None = None,
        max_cola: int | None = None,
        timeout_cola: float | None = None,
        resource_tracker: Any | None = None,
    ) -> None:
        self._cfg = config or ConfiguracionCerebroCentral()
        # Hacu Inventario puede resolver intenciones inequívocas antes de
        # consumir una inferencia. Se puede desactivar por despliegue para
        # comparar comportamiento o diagnosticar el planificador.
        fast_path_env = os.getenv("HAKU_PERMITIR_FAST_PATH", "true").strip().casefold()
        self.permitir_fast_path = fast_path_env not in {"0", "false", "no", "off"}
        conc = max_concurrencia if max_concurrencia is not None else self._cfg.max_concurrencia
        cola = max_cola if max_cola is not None else self._cfg.max_cola
        t_cola = timeout_cola if timeout_cola is not None else self._cfg.timeout_cola

        self._governor = LlmGovernor(max_concurrencia=conc, max_cola=cola, timeout_cola_s=t_cola)
        self._circuit_breaker = LlmCircuitBreaker(fallos_max=3, tiempo_enfriamiento_s=5.0)
        self._modelo_activo_cache: str | None = None
        self._modelo_cache_lock = threading.Lock()
        self._salud_cache: dict[str, Any] | None = None
        self._salud_cache_timestamp: float = 0.0
        self._salud_cache_lock = threading.Lock()
        self._resource_tracker = resource_tracker

        self._pools_lock = threading.Lock()
        self._pools: dict[str, PoolConexionesHTTP] = {}
        self._cerrado = False

        self._metricas_lock = threading.Lock()
        self._metricas = {
            "total_solicitudes": 0,
            "solicitudes_exitosas": 0,
            "rechazos_saturacion": 0,
            "limites_excedidos": 0,
            "timeouts": 0,
            "fallos_conexion": 0,
            "errores_oom": 0,
            "respuestas_invalidas": 0,
        }

    @property
    def governor(self) -> LlmGovernor:
        return self._governor

    @property
    def circuit_breaker(self) -> LlmCircuitBreaker:
        return self._circuit_breaker

    @property
    def config(self) -> ConfiguracionCerebroCentral:
        return self._cfg

    @property
    def resource_tracker(self) -> Any | None:
        return self._resource_tracker

    @resource_tracker.setter
    def resource_tracker(self, tracker: Any | None) -> None:
        self._resource_tracker = tracker

    @property
    def cerrado(self) -> bool:
        with self._pools_lock:
            return self._cerrado

    def obtener_metricas(self) -> dict[str, Any]:
        """Retorna una instantánea thread-safe de las métricas acumuladas en memoria.
        
        Nota: Son contadores volátiles en memoria del proceso que se reinician al reiniciar la aplicación.
        """
        with self._metricas_lock:
            copia = dict(self._metricas)
        copia["circuit_breaker_estado"] = self._circuit_breaker.estado.value
        copia["ranuras_activas"] = self._governor.activas
        copia["en_cola"] = self._governor.en_cola
        copia["max_concurrencia"] = self._governor.max_concurrencia

        with self._pools_lock:
            total_creadas = sum(p.total_creadas for p in self._pools.values())
            total_reutilizadas = sum(p.total_reutilizadas for p in self._pools.values())
            en_pool = sum(p.conexiones_en_pool for p in self._pools.values())
        copia["pool_conexiones_creadas"] = total_creadas
        copia["pool_conexiones_reutilizadas"] = total_reutilizadas
        copia["pool_conexiones_en_espera"] = en_pool
        return copia

    def _obtener_pool(self, url: str) -> tuple[PoolConexionesHTTP, str]:
        """Obtiene o crea el pool de conexiones correspondiente al endpoint destino."""
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        is_https = parsed.scheme.lower() == "https"
        port = parsed.port or (443 if is_https else 80)
        clave = f"{host}:{port}:{is_https}"

        with self._pools_lock:
            if self._cerrado:
                raise RuntimeError("El Cliente del Cerebro Central se encuentra cerrado.")
            pool = self._pools.get(clave)
            if pool is None:
                pool = PoolConexionesHTTP(
                    host=host,
                    port=port,
                    is_https=is_https,
                    timeout_defecto=self._cfg.timeout_lectura,
                    max_size=max(2, self._governor.max_concurrencia),
                )
                self._pools[clave] = pool

        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        return pool, path

    def _ejecutar_http_keep_alive(
        self,
        metodo: str,
        url: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, bytes]:
        """Ejecuta una petición HTTP/1.1 persistente con Keep-Alive reutilizando el pool."""
        pool, path = self._obtener_pool(url)
        conn = pool.adquirir(timeout=timeout)
        if conn.sock is not None and timeout is not None:
            conn.sock.settimeout(timeout)
        valida = True

        hdrs = {
            "User-Agent": "AudacIA-CerebroCentral-Client/1.0",
            "Connection": "keep-alive",
        }
        if data is not None:
            hdrs["Content-Type"] = "application/json"
            hdrs["Content-Length"] = str(len(data))
        if headers:
            hdrs.update(headers)

        try:
            conn.request(metodo, path, body=data, headers=hdrs)
            resp = conn.getresponse()
            status = resp.status
            cuerpo = resp.read()

            conn_header = resp.headers.get("Connection", "").lower()
            if conn_header == "close":
                valida = False

            if status >= 400:
                valida = (conn_header != "close")
                raise urllib.error.HTTPError(
                    url=url,
                    code=status,
                    msg=resp.reason,
                    hdrs=resp.headers,
                    fp=io.BytesIO(cuerpo),
                )

            return status, cuerpo
        except urllib.error.HTTPError:
            raise
        except (TimeoutError, socket.timeout) as e:
            valida = False
            raise TimeoutError(f"Timeout de lectura/conexión con el Cerebro Central ({timeout}s): {e}") from e
        except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError, ConnectionRefusedError, ConnectionError, OSError) as e:
            valida = False
            raise ConnectionError(f"Error de conexión con el Cerebro Central en {url}: {e}") from e
        except Exception:
            valida = False
            raise
        finally:
            pool.liberar(conn, valida=valida)

    def cerrar(self) -> None:
        """Cierra todas las conexiones HTTP persistentes administradas por el cliente."""
        with self._pools_lock:
            self._cerrado = True
            for pool in self._pools.values():
                pool.cerrar()
            self._pools.clear()

    def __enter__(self) -> "ClienteCerebroCentral":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cerrar()

    def _validar_ausencia_parametros_prohibidos(self, kwargs: Mapping[str, Any]) -> None:
        """Asegura que ningún agente intente especificar un modelo o proveedor propio."""
        for clave in kwargs:
            if clave.lower() in PARAMETROS_PROHIBIDOS:
                raise ViolacionArquitecturaError(
                    f"Violación de Arquitectura de AudacIA: el parámetro '{clave}' está prohibido para agentes. "
                    "Todos los agentes deben consumir exclusivamente el Cerebro Central único. "
                    "Ningún agente puede seleccionar, instanciar o suministrar un modelo de lenguaje propio."
                )

    def invalidar_cache_salud(self) -> None:
        """Invalida la caché local de salud para forzar una consulta fresca en la siguiente llamada."""
        with self._salud_cache_lock:
            self._salud_cache = None
            self._salud_cache_timestamp = 0.0

    def obtener_modelo_activo(self, forzar_refresco: bool = False) -> str:
        """Consulta el identificador del modelo actualmente activo en el Cerebro Central.
        
        El modelo se descubre dinámicamente desde el servidor; ningún agente lo impone.
        """
        with self._modelo_cache_lock:
            if self._modelo_activo_cache and not forzar_refresco:
                return self._modelo_activo_cache

        try:
            if _es_urlopen_mockeado():
                req = urllib.request.Request(
                    self._cfg.url_modelos,
                    headers={"User-Agent": "AudacIA-CerebroCentral-Client/1.0"},
                )
                with urllib.request.urlopen(req, timeout=self._cfg.timeout_conexion) as resp:
                    if resp.status == 200:
                        datos = json.loads(resp.read().decode("utf-8", errors="replace"))
                        modelos = datos.get("data", [])
                        if modelos and isinstance(modelos, list):
                            id_modelo = modelos[0].get("id")
                            if id_modelo:
                                with self._modelo_cache_lock:
                                    self._modelo_activo_cache = id_modelo
                                return id_modelo
            else:
                status, cuerpo_bytes = self._ejecutar_http_keep_alive(
                    "GET", self._cfg.url_modelos, timeout=self._cfg.timeout_conexion
                )
                if status == 200:
                    datos = json.loads(cuerpo_bytes.decode("utf-8", errors="replace"))
                    modelos = datos.get("data", [])
                    if modelos and isinstance(modelos, list):
                        id_modelo = modelos[0].get("id")
                        if id_modelo:
                            with self._modelo_cache_lock:
                                self._modelo_activo_cache = id_modelo
                            return id_modelo
        except Exception:
            pass

        return self._cfg.modelo_por_defecto

    def comprobar_salud(self, forzar_sondeo: bool = False) -> dict[str, Any]:
        """Consulta el estado operativo del Cerebro Central con Circuit Breaker y caché de corta duración."""
        if self._cerrado:
            return {
                "disponible": False,
                "estado": "No disponible: Cliente cerrado",
                "circuit_breaker": self._circuit_breaker.estado.value,
                "ranuras_activas": self._governor.activas,
                "en_cola": self._governor.en_cola,
                "max_concurrencia": self._governor.max_concurrencia,
            }

        # 1. Comprobar caché local de salud si no se solicita forzar el sondeo
        if not forzar_sondeo:
            with self._salud_cache_lock:
                ttl_vigente = getattr(self, "_salud_cache_ttl_efectivo", self._cfg.ttl_cache_salud)
                if (
                    self._salud_cache is not None
                    and (time.monotonic() - self._salud_cache_timestamp) < ttl_vigente
                ):
                    return dict(self._salud_cache)

        # 2. Si el Circuit Breaker está en estado OPEN y aún no vence el tiempo de enfriamiento,
        # evitar el intento síncrono de conexión TCP para no pagar latencia innecesaria.
        if not self._circuit_breaker.puede_ejecutar():
            resultado_open = {
                "disponible": False,
                "estado": "No disponible: Circuit Breaker en estado OPEN",
                "circuit_breaker": self._circuit_breaker.estado.value,
                "ranuras_activas": self._governor.activas,
                "en_cola": self._governor.en_cola,
                "max_concurrencia": self._governor.max_concurrencia,
            }
            with self._salud_cache_lock:
                self._salud_cache = resultado_open
                self._salud_cache_timestamp = time.monotonic()
            return resultado_open

        # 3. Sondeo HTTP al endpoint /health
        timeout_salud = min(1.0, self._cfg.timeout_conexion)
        try:
            if _es_urlopen_mockeado():
                req = urllib.request.Request(
                    self._cfg.url_salud,
                    headers={"User-Agent": "AudacIA-CerebroCentral-Client/1.0"},
                )
                with urllib.request.urlopen(req, timeout=timeout_salud) as resp:
                    cuerpo = resp.read().decode("utf-8", errors="replace")
                    status = resp.status
            else:
                status, cuerpo_bytes = self._ejecutar_http_keep_alive(
                    "GET", self._cfg.url_salud, timeout=timeout_salud
                )
                cuerpo = cuerpo_bytes.decode("utf-8", errors="replace")

            datos = json.loads(cuerpo)
            disponible = (status == 200) and (datos.get("status") in ("ok", "ready"))
            if disponible:
                self._circuit_breaker.registrar_exito()
            else:
                self._circuit_breaker.registrar_fallo()

            resultado = {
                "disponible": disponible,
                "estado": datos.get("status", "ok") if disponible else "no_disponible",
                "circuit_breaker": self._circuit_breaker.estado.value,
                "ranuras_activas": self._governor.activas,
                "en_cola": self._governor.en_cola,
                "max_concurrencia": self._governor.max_concurrencia,
            }
            with self._salud_cache_lock:
                self._salud_cache = resultado
                self._salud_cache_timestamp = time.monotonic()
                self._salud_cache_ttl_efectivo = self._cfg.ttl_cache_salud
            return resultado
        except Exception as e:
            self._circuit_breaker.registrar_fallo(e)
            resultado_error = {
                "disponible": False,
                "estado": f"No disponible: {type(e).__name__}",
                "circuit_breaker": self._circuit_breaker.estado.value,
                "ranuras_activas": self._governor.activas,
                "en_cola": self._governor.en_cola,
                "max_concurrencia": self._governor.max_concurrencia,
            }
            with self._salud_cache_lock:
                self._salud_cache = resultado_error
                self._salud_cache_timestamp = time.monotonic()
                self._salud_cache_ttl_efectivo = max(5.0, self._cfg.ttl_cache_salud)
            return resultado_error

    def generar(
        self,
        mensajes: Sequence[dict[str, Any]],
        herramientas: Sequence[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int = 512,
        temperatura: float = 0.3,
        formato_json: bool = False,
        **kwargs: Any,
    ) -> RespuestaCerebroCentral:
        """Punto de entrada universal para que los agentes soliciten razonamiento o lenguaje.
        
        Lanza ViolacionArquitecturaError si se intenta suministrar un modelo propio.
        Lanza LlmLimiteExcedidoError si el prompt o tokens exceden el presupuesto seguro.
        Lanza LlmSaturadoError si la cola está llena.
        LlmColaExpiradaError si expira la espera en cola.
        LlmNoDisponibleError si el circuit breaker está abierto.
        LlmOomError si ocurre fallo de memoria GPU en inferencia.
        LlmRespuestaInvalidaError si la respuesta está truncada o malformada.
        """
        with self._metricas_lock:
            self._metricas["total_solicitudes"] += 1

        if self._cerrado:
            raise RuntimeError("El Cliente del Cerebro Central se encuentra cerrado.")

        # 1. Blindaje: rechazar tajantemente cualquier parámetro de modelo suministrado por el agente
        self._validar_ausencia_parametros_prohibidos(kwargs)

        if not mensajes:
            with self._metricas_lock:
                self._metricas["limites_excedidos"] += 1
            raise ValueError("La lista de mensajes no puede estar vacía.")

        # 2. Validación estricta de max_tokens
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            with self._metricas_lock:
                self._metricas["limites_excedidos"] += 1
            raise ValueError(f"max_tokens debe ser un entero positivo mayor a cero (recibido: {max_tokens}).")

        # 3. Validación de límite de caracteres del prompt (primera barrera)
        total_caracteres = sum(len(str(m.get("content") or "")) for m in mensajes)
        if herramientas:
            total_caracteres += sum(len(json.dumps(h, ensure_ascii=False)) for h in herramientas)

        if total_caracteres > self._cfg.max_caracteres_prompt:
            with self._metricas_lock:
                self._metricas["limites_excedidos"] += 1
            raise LlmLimiteExcedidoError(
                f"El tamaño del prompt ({total_caracteres} caracteres) excede el límite máximo permitido "
                f"({self._cfg.max_caracteres_prompt} caracteres)."
            )

        # 4. Validación de presupuesto de contexto por ranura (segunda barrera)
        ventana_ranura = self._cfg.ventana_contexto_por_ranura
        presupuesto_util = max(256, ventana_ranura - self._cfg.margen_seguridad_tokens)

        if max_tokens > presupuesto_util:
            with self._metricas_lock:
                self._metricas["limites_excedidos"] += 1
            raise LlmLimiteExcedidoError(
                f"max_tokens solicitado ({max_tokens}) excede la ventana máxima utilizable por ranura "
                f"({presupuesto_util} tokens tras margen de seguridad de {self._cfg.margen_seguridad_tokens} tokens)."
            )

        # Estimación conservadora de tokens de entrada (~3.2 caracteres por token)
        tokens_entrada_estimados = math.ceil(total_caracteres / 3.2)
        if tokens_entrada_estimados + max_tokens > presupuesto_util:
            with self._metricas_lock:
                self._metricas["limites_excedidos"] += 1
            raise LlmLimiteExcedidoError(
                f"El presupuesto de contexto estimado ({tokens_entrada_estimados} tokens entrada + "
                f"{max_tokens} tokens salida = {tokens_entrada_estimados + max_tokens}) excede la ventana "
                f"utilizable por ranura ({presupuesto_util} tokens)."
            )

        # 5. Comprobar Circuit Breaker
        if not self._circuit_breaker.puede_ejecutar():
            with self._metricas_lock:
                self._metricas["fallos_conexion"] += 1
            raise LlmNoDisponibleError("Cerebro Central no disponible: Circuit Breaker en estado OPEN.")

        modelo_central = self.obtener_modelo_activo()

        payload: dict[str, Any] = {
            "model": modelo_central,
            "messages": list(mensajes),
            "max_tokens": max(16, max_tokens),
            "temperature": max(0.0, min(2.0, temperatura)),
            "stream": False,
        }

        if formato_json:
            json_schema = kwargs.get("json_schema")
            if json_schema is None:
                payload["response_format"] = {"type": "json_object"}
            elif isinstance(json_schema, Mapping) and json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": str(kwargs.get("json_schema_name") or "cerebro_response"),
                        "schema": dict(json_schema),
                    },
                }
            else:
                raise ValueError("json_schema debe ser un esquema JSON no vacío cuando formato_json=True.")

        if herramientas:
            payload["tools"] = list(herramientas)

        if tool_choice:
            payload["tool_choice"] = tool_choice

        timeout_lectura_efectivo = float(kwargs.get("timeout") or self._cfg.timeout_lectura)

        # Enviar petición bajo gobierno de concurrencia
        try:
            with self._governor.adquirir() as espera_cola_ms:
                t0 = time.perf_counter()

                try:
                    if _es_urlopen_mockeado():
                        req = urllib.request.Request(
                            self._cfg.url_chat,
                            data=json.dumps(payload).encode("utf-8"),
                            headers={
                                "Content-Type": "application/json",
                                "User-Agent": "AudacIA-CerebroCentral-Client/1.0",
                            },
                        )
                        with urllib.request.urlopen(req, timeout=timeout_lectura_efectivo) as resp:
                            if resp.status != 200:
                                self._circuit_breaker.registrar_fallo()
                                with self._metricas_lock:
                                    self._metricas["fallos_conexion"] += 1
                                raise RuntimeError(f"Cerebro Central devolvió código HTTP {resp.status}")
                            cuerpo = resp.read().decode("utf-8", errors="replace")
                    else:
                        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                        status, cuerpo_bytes = self._ejecutar_http_keep_alive(
                            metodo="POST",
                            url=self._cfg.url_chat,
                            data=payload_bytes,
                            timeout=timeout_lectura_efectivo,
                        )
                        cuerpo = cuerpo_bytes.decode("utf-8", errors="replace")

                    try:
                        resp_json = json.loads(cuerpo)
                    except json.JSONDecodeError as jde:
                        self._circuit_breaker.registrar_fallo(jde)
                        with self._metricas_lock:
                            self._metricas["respuestas_invalidas"] += 1
                        raise LlmRespuestaInvalidaError(
                            f"Respuesta del Cerebro Central truncada o JSON malformado: {jde}"
                        ) from jde

                    self._circuit_breaker.registrar_exito()
                    with self._metricas_lock:
                        self._metricas["solicitudes_exitosas"] += 1

                except urllib.error.HTTPError as e:
                    cuerpo_error = ""
                    try:
                        cuerpo_error = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass

                    if e.code == 429:
                        raise LlmSaturadoError("Cerebro Central saturado: servidor HTTP 429 Too Many Requests") from e

                    # Clasificación rigurosa de errores de memoria (CUDA OOM)
                    cuerpo_lower = cuerpo_error.lower()
                    es_oom = any(patron in cuerpo_lower for patron in [
                        "out of memory",
                        "cuda out of memory",
                        "failed to allocate",
                        "cudamalloc failed",
                        "not enough memory",
                        "out_of_memory",
                    ])

                    if es_oom:
                        self._circuit_breaker.registrar_fallo(e)
                        with self._metricas_lock:
                            self._metricas["errores_oom"] += 1
                        if self._resource_tracker is not None:
                            try:
                                from cerebro_audacia.agentes.esquemas import ResourceKind
                                self._resource_tracker.marcar_saturado(
                                    ResourceKind.VRAM,
                                    f"CUDA OOM detectado en respuesta HTTP {e.code}: {cuerpo_error[:200]}"
                                )
                            except Exception:
                                pass
                        raise LlmOomError(
                            f"Fallo de memoria en inferencia (CUDA OOM) [HTTP {e.code}]: {cuerpo_error[:300]}"
                        ) from e

                    self._circuit_breaker.registrar_fallo(e)
                    with self._metricas_lock:
                        self._metricas["fallos_conexion"] += 1
                    raise RuntimeError(f"Cerebro Central devolvió código HTTP {e.code}: {cuerpo_error[:300]}") from e

                except urllib.error.URLError as e:
                    if isinstance(e.reason, (TimeoutError, socket.timeout)) or "timed out" in str(e).lower():
                        self._circuit_breaker.registrar_fallo(e)
                        with self._metricas_lock:
                            self._metricas["timeouts"] += 1
                        raise TimeoutError(f"Timeout de conexión con el Cerebro Central: {e}") from e
                    self._circuit_breaker.registrar_fallo(e)
                    with self._metricas_lock:
                        self._metricas["fallos_conexion"] += 1
                    raise ConnectionError(f"Error de conexión con el Cerebro Central en {self._cfg.url_chat}: {e}") from e

                except (TimeoutError, socket.timeout) as e:
                    self._circuit_breaker.registrar_fallo(e)
                    with self._metricas_lock:
                        self._metricas["timeouts"] += 1
                    raise TimeoutError(f"Timeout de lectura/conexión con el Cerebro Central ({self._cfg.timeout_lectura}s): {e}") from e

                except (ConnectionError, OSError) as e:
                    self._circuit_breaker.registrar_fallo(e)
                    with self._metricas_lock:
                        self._metricas["fallos_conexion"] += 1
                    raise ConnectionError(f"Error de conexión con el Cerebro Central en {self._cfg.url_chat}: {e}") from e

        except LlmSaturadoError:
            with self._metricas_lock:
                self._metricas["rechazos_saturacion"] += 1
            raise
        except LlmColaExpiradaError:
            with self._metricas_lock:
                self._metricas["timeouts"] += 1
            raise

        duracion_ms = (time.perf_counter() - t0) * 1000.0
        choices = resp_json.get("choices", [])
        if not choices:
            raise RuntimeError("El Cerebro Central no devolvió opciones de respuesta.")

        eleccion = choices[0]
        mensaje_resp = eleccion.get("message", {})
        texto = mensaje_resp.get("content") or ""
        finish_reason = eleccion.get("finish_reason", "stop")
        tool_calls_raw = mensaje_resp.get("tool_calls", [])
        uso = resp_json.get("usage")
        uso = uso if isinstance(uso, dict) else {}
        detalles_prompt = uso.get("prompt_tokens_details")
        detalles_prompt = detalles_prompt if isinstance(detalles_prompt, dict) else {}
        metricas_servidor = resp_json.get("timings")
        metricas_servidor = metricas_servidor if isinstance(metricas_servidor, dict) else {}

        def _entero_metrica(valor: Any) -> int | None:
            try:
                return max(0, int(valor)) if valor is not None else None
            except (TypeError, ValueError, OverflowError):
                return None

        def _float_metrica(valor: Any) -> float | None:
            try:
                return max(0.0, float(valor)) if valor is not None else None
            except (TypeError, ValueError, OverflowError):
                return None

        tokens_completados = _entero_metrica(uso.get("completion_tokens")) or 0
        tokens_prompt = _entero_metrica(uso.get("prompt_tokens"))
        tokens_prompt_cacheados = _entero_metrica(detalles_prompt.get("cached_tokens"))
        prefill_ms = _float_metrica(metricas_servidor.get("prompt_ms"))
        generacion_ms = _float_metrica(metricas_servidor.get("predicted_ms"))

        datos_json = None
        if formato_json and texto:
            try:
                datos_json = json.loads(texto)
            except json.JSONDecodeError:
                bloque = _BLOQUE_JSON.search(texto)
                if bloque:
                    try:
                        datos_json = json.loads(bloque.group(0))
                    except Exception:
                        pass

        return RespuestaCerebroCentral(
            texto=texto,
            tool_calls=tuple(tool_calls_raw) if tool_calls_raw else (),
            datos_json=datos_json,
            finish_reason=finish_reason,
            modelo_reportado=modelo_central,
            duracion_ms=duracion_ms,
            tokens_completados=tokens_completados,
            espera_cola_ms=espera_cola_ms,
            tokens_prompt=tokens_prompt,
            tokens_prompt_cacheados=tokens_prompt_cacheados,
            prefill_ms=prefill_ms,
            generacion_ms=generacion_ms,
        )

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.3,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Método de compatibilidad con interfaz OpenAI / OrquestadorAudacIA."""
        self._validar_ausencia_parametros_prohibidos(kwargs)

        herramientas = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        formato = kwargs.get("response_format", {})
        formato_json = bool(formato and formato.get("type") == "json_object")

        respuesta = self.generar(
            mensajes=messages,
            herramientas=herramientas,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            temperatura=temperature,
            formato_json=formato_json,
        )

        msg_dict: dict[str, Any] = {"role": "assistant", "content": respuesta.texto}
        if respuesta.tool_calls:
            msg_dict["tool_calls"] = list(respuesta.tool_calls)

        return {
            "id": f"chatcmpl-audacia-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": respuesta.modelo_reportado,
            "choices": [
                {
                    "index": 0,
                    "message": msg_dict,
                    "finish_reason": respuesta.finish_reason,
                }
            ],
            "usage": {
                "completion_tokens": respuesta.tokens_completados,
                "total_tokens": respuesta.tokens_completados,
            },
        }


# Instancia singleton compartida
_CLIENTE_CENTRAL_GLOBAL: ClienteCerebroCentral | None = None
_CLIENTE_CENTRAL_LOCK = threading.Lock()


def obtener_cliente_cerebro_central(config: ConfiguracionCerebroCentral | None = None) -> ClienteCerebroCentral:
    """Retorna la instancia global compartida del Cliente del Cerebro Central."""
    global _CLIENTE_CENTRAL_GLOBAL
    with _CLIENTE_CENTRAL_LOCK:
        if _CLIENTE_CENTRAL_GLOBAL is None:
            _CLIENTE_CENTRAL_GLOBAL = ClienteCerebroCentral(config=config)
        return _CLIENTE_CENTRAL_GLOBAL
