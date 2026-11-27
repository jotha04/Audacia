# Cerebro AudacIA

Núcleo local de inferencia y gobernanza de modelos de lenguaje para **AudacIA (Universidad Simón Bolívar)**.

## Componentes

| Directorio | Responsabilidad |
|---|---|
| `agentes/` | Contratos formales (`contrato.py`), esquemas, orquestador y políticas de resiliencia. |
| `conexion/` | Cliente HTTP único (`ClienteCerebroCentral`) con pool keep-alive, cola de concurrencia y *circuit breaker*. |
| `proveedor_modelo/` | Configuración (`config_llm.json`, perfil `servidor_12b` por defecto), scripts de ciclo de vida de `llama-server` y descargadores de binarios. |
| `conocimiento/` | Lineamientos de fuentes documentales. |

## Arranque

- **Automático (recomendado):** ejecutar `iniciar_hacu_inventario.cmd` en la raíz del repositorio.
- **Manual (solo El Cerebro):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File cerebro_audacia\proveedor_modelo\iniciar_llama_server.ps1
  ```
- **Verificar estado:** `http://127.0.0.1:8080/health`
