# Cerebro AudacIA — Hacu Inventario & Hacu Visitante

Plataforma local de inteligencia artificial de **AudacIA (Universidad Simón Bolívar)** que centraliza la inferencia de lenguaje natural en un único servicio local (**El Cerebro**) y aloja dos aplicaciones:

1. **El Cerebro (`cerebro_audacia/`)** — Proveedor único de inferencia local basado en `llama-server` (API OpenAI-compatible en `http://127.0.0.1:8080`), con control de concurrencia, colas y *circuit breaker*.
2. **Hacu Inventario (`sistema_inventario/`)** — Sistema web de gestión de inventario en Django con control de permisos granulares (RBAC), trazabilidad de movimientos, préstamos, devoluciones, traslados, conteos físicos, reportes/exportaciones y asistente conversacional por voz y texto conectado al Cerebro.
3. **Hacu Visitante (`hacu/` + `run_hacu.py`)** — Asistente expositor conversacional con RAG sobre los proyectos de AudacIA y la Universidad Simón Bolívar.

> **Nota de arquitectura sobre Hacu Visitante:**  
> **No se realizó una conexión ni integración directa entre Hacu Visitante y la plataforma web de Hacu Inventario.** Hacu Visitante funciona de forma independiente mediante su propio lanzador ([`iniciar_hacu_visitante.cmd`](iniciar_hacu_visitante.cmd) o `run_hacu.py`).

---

## Modelo LLM utilizado (Llama 12B)

El perfil por defecto del Cerebro (`servidor_12b` en [`cerebro_audacia/proveedor_modelo/config_llm.json`](cerebro_audacia/proveedor_modelo/config_llm.json)) está configurado para trabajar con un modelo **Llama 12B (`.gguf`)** descargado en el equipo destino:

- Coloca el archivo `.gguf` de **Llama 12B** dentro de la carpeta **`models/`** (esta carpeta existe en el repositorio pero los pesos `.gguf` están excluidos en `.gitignore` por su tamaño).
- El lanzador detecta automáticamente cualquier archivo `*12b*.gguf` dentro de `models/` (o el `.gguf` disponible) y selecciona automáticamente `CUDA0` si hay GPU NVIDIA o `Vulkan0` en caso contrario.

---

## Inicio rápido en Windows (un solo clic)

### 1. El Cerebro + Servidor Hacu Inventario (Modo Servidor LAN con IP fija)

Haz doble clic en **[`iniciar_hacu_inventario.cmd`](iniciar_hacu_inventario.cmd)** (o ejecútalo desde PowerShell/CMD):

```cmd
iniciar_hacu_inventario.cmd
```

*(Opcional: fija la IP en `sistema_inventario/.env` con `SERVER_IP=172.16.x.x` o pásala como parámetro `iniciar_hacu_inventario.cmd -IpFija 172.16.x.x`. Si se deja vacío, el lanzador autodetecta la IP fija del equipo.)*

¿Qué hace automáticamente?
1. Usa Python 3.12 de 64 bits para el entorno virtual `.venv`. Si encuentra un entorno incompatible, lo conserva como `.venv-anterior-*` antes de crear uno nuevo.
2. Comprueba si las dependencias están instaladas; si faltan, instala automáticamente `sistema_inventario/requirements-django.txt` (incluyendo **Uvicorn ASGI** y **Cryptography** para HTTPS) y `sistema_inventario/requirements-haku-voz.txt`.
3. Genera `sistema_inventario/.env` (con clave secreta aleatoria) si no existe y aplica las migraciones de base de datos.
4. Genera/valida automáticamente los **certificados TLS (HTTPS)** incluyendo `localhost`, `127.0.0.1` y la **IP fija LAN del computador** (indispensable para que otros PCs y celulares puedan usar el micrófono en Hacu Voz).
5. Comprueba que el modelo `.gguf` esté en `models/` y descarga los binarios de `llama-server.exe` si faltan.
6. Levanta **El Cerebro (`http://127.0.0.1:8080`, interno en el servidor)** y el **Servidor Hacu Inventario (`https://0.0.0.0:8000`)** accesible por la IP fija del equipo (`https://<IP_FIJA>:8000/`).
7. Cuando ambos servicios están operativos, muestra el aviso de listo y **abre automáticamente el navegador** en `https://<IP_FIJA>:8000/`.

**Primer arranque en el servidor (una sola vez):**
- Crear superusuario:
  ```powershell
  .\.venv\Scripts\python.exe sistema_inventario\manage.py createsuperuser
  ```
- Abrir el puerto `8000` en el Firewall de Windows (PowerShell como Administrador) para que otros equipos de la red puedan conectarse a `https://<IP_FIJA>:8000/`:
  ```powershell
  netsh advfirewall firewall add rule name="Hacu Inventario HTTPS" dir=in action=allow protocol=TCP localport=8000
  ```
- *(Opcional, para quitar el aviso de certificado autofirmado en navegadores de la red)*: instalar `mkcert` (`winget install FiloSottile.mkcert`) antes de ejecutar `iniciar_hacu_inventario.cmd`, o aceptar la excepción la primera vez en cada cliente.

### 2. Hacu Visitante (independiente)

Para iniciar Hacu Visitante por separado, ejecuta **[`iniciar_hacu_visitante.cmd`](iniciar_hacu_visitante.cmd)**:

```cmd
iniciar_hacu_visitante.cmd
```

También acepta los argumentos propios de `run_hacu.py`:

```cmd
iniciar_hacu_visitante.cmd --ui --voz
iniciar_hacu_visitante.cmd --ui --voz --pantalla-completa
```

*(Instala sus dependencias en un entorno aislado `.venv_visitante` si faltan, asegura que `llama-server` esté disponible en el puerto `8080` y arranca `run_hacu.py`.)*

---

## Estructura del repositorio

```text
cerebro-audacia/
├── iniciar_hacu_inventario.cmd      # Lanzador automático: instala deps, inicia Cerebro + Hacu Inventario y abre el navegador
├── iniciar_hacu_visitante.cmd       # Lanzador independiente para Hacu Visitante
├── README.md                        # Documentación general del proyecto
├── .gitignore                       # Exclusión de entornos, secretos, BD locales, binarios y pesos GGUF
├── Dockerfile                       # Imagen opcional con GPU para Hacu Visitante
├── requirements.txt                 # Dependencias base de Hacu Visitante (RAG)
├── requirements-ui.txt              # Interfaz gráfica opcional de Hacu Visitante (PySide6)
├── requirements-voz.txt             # Voz opcional de Hacu Visitante
├── run_hacu.py                      # Punto de entrada CLI/UI de Hacu Visitante
├── scripts/                         # Scripts PowerShell de arranque y despliegue
│   ├── iniciar_hacu_inventario.ps1
│   ├── iniciar_hacu_visitante.ps1
│   └── iniciar_robot_docker.ps1
├── cerebro_audacia/                 # El Cerebro Central (cliente LLM único, orquestación y proveedor llama-server)
│   ├── agentes/
│   ├── conexion/
│   ├── conocimiento/
│   └── proveedor_modelo/            # config_llm.json (perfil servidor_12b), scripts de inicio/salud y descargadores
├── sistema_inventario/              # Hacu Inventario (Django + Web + Voz/Chat + RBAC + Reportes)
│   ├── .env.example
│   ├── manage.py
│   ├── requirements-django.txt
│   ├── requirements-haku-voz.txt
│   ├── cuentas/                     # Usuarios, roles y permisos granulares (RBAC)
│   ├── inventario/                  # Modelos, servicios transaccionales, catálogo, movimientos, préstamos y conteos
│   ├── integraciones/haku/          # Asistente conversacional/voz, normalizador, planificador y TTS/STT
│   ├── web/                         # Dashboard analítico, reportes y exportaciones CSV/Excel
│   ├── scripts/                     # Utilidades de respaldos y despliegue HTTPS
│   ├── static/
│   └── templates/
├── hacu/                            # Núcleo de Hacu Visitante (RAG, memoria, estilo, voz e interfaz)
├── documents/                       # Corpus documental de AudacIA y Universidad Simón Bolívar
├── herramientas/                    # Utilidades de construcción de corpus y cálculo de VRAM
└── models/                          # Carpeta donde se coloca el modelo Llama 12B (.gguf)
```

---

## Requirements / Dependencias

El proyecto incluye todos los archivos `requirements*.txt` para instalación automática (mediante los lanzadores `.cmd`) o manual con `pip`:

- **Hacu Inventario (Django + ASGI HTTPS):** [`sistema_inventario/requirements-django.txt`](sistema_inventario/requirements-django.txt)
- **Hacu Inventario — Voz (STT faster-whisper + TTS Piper):** [`sistema_inventario/requirements-haku-voz.txt`](sistema_inventario/requirements-haku-voz.txt)
- **Hacu Visitante (núcleo RAG):** [`requirements.txt`](requirements.txt)
- **Hacu Visitante — Voz opcional:** [`requirements-voz.txt`](requirements-voz.txt)
- **Hacu Visitante — Interfaz gráfica opcional (PySide6):** [`requirements-ui.txt`](requirements-ui.txt)

Instalación manual de referencia (dentro de un entorno virtual Python 3.10–3.12):

```powershell
# Hacu Inventario (Web + Voz)
pip install -r sistema_inventario/requirements-django.txt
pip install -r sistema_inventario/requirements-haku-voz.txt

# Hacu Visitante (RAG + Voz + UI opcionales)
pip install -r requirements.txt
pip install -r requirements-voz.txt
pip install -r requirements-ui.txt
```

---

## Requisitos del equipo destino

- **Sistema operativo:** Windows 10/11 (64 bits) con PowerShell 5.1+
- **Python:** 3.10 a 3.12 (agregado al `PATH`)
- **GPU recomendada:** NVIDIA con soporte CUDA (≥ 12 GB VRAM para Llama 12B Q4_K_M; funciona también con 8 GB usando modelos 8B o perfiles alternos en `config_llm.json`)
- **Espacio en disco:** ~15 GB libres (modelo GGUF 12B + dependencias + voces STT/TTS)

---

## Operación manual útil

- **Detener El Cerebro:** `powershell -ExecutionPolicy Bypass -File cerebro_audacia\proveedor_modelo\detener_llama_server.ps1`
- **Verificar salud de El Cerebro:** `powershell -ExecutionPolicy Bypass -File cerebro_audacia\proveedor_modelo\comprobar_salud_llm.ps1`
- **Cambiar de perfil LLM:** `iniciar_hacu_inventario.cmd -PerfilLLM servidor_rtx5070ti` (o `desarrollo_rtx2000ada`, `desarrollo_rx7600`)
- **Variables de entorno de Hacu Inventario:** consultar [`sistema_inventario/.env.example`](sistema_inventario/.env.example). Nunca versionar `.env`, `db.sqlite3` ni archivos `.gguf`.
