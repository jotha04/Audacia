# ==============================================================================
# AUDACIA - SCRIPT DE INICIO DE LLAMA-SERVER (INFERENCIA LOCAL)
# ==============================================================================

param(
    [string]$Perfil = "",
    [string]$ModeloOverride = "",
    [string]$BinarioOverride = "",
    [int]$PuertoOverride = 0,
    [switch]$Esperar
)

$ErrorActionPreference = "Stop"

$DirectorioScript = $PSScriptRoot
$RaizProyecto = (Resolve-Path "$DirectorioScript\..\..").Path
$ArchivoConfig = Join-Path $DirectorioScript "config_llm.json"
$ArchivoLog = Join-Path $DirectorioScript "llama_server.log"
$ArchivoErr = Join-Path $DirectorioScript "llama_server_err.log"

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " AudacIA - Iniciando Servicio LLM Local (llama-server)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

# 1. Cargar Configuración
if (-not (Test-Path $ArchivoConfig)) {
    Write-Error "ERROR: No se encontró el archivo de configuración en: $ArchivoConfig"
    exit 1
}

$ConfigRaw = Get-Content -Path $ArchivoConfig -Raw -Encoding UTF8 | ConvertFrom-Json
$HostLocal = $ConfigRaw.servicio.host
$Puerto = if ($PuertoOverride -gt 0) { $PuertoOverride } else { [int]$ConfigRaw.servicio.puerto }
$UrlSalud = "http://" + $HostLocal + ":" + $Puerto + "/health"

if ([string]::IsNullOrWhiteSpace($Perfil)) { $Perfil = $env:AUDACIA_LLM_PERFIL }
if ([string]::IsNullOrWhiteSpace($Perfil)) { $Perfil = $env:AUDACIA_LLM_PROFILE }
if ([string]::IsNullOrWhiteSpace($Perfil)) { $Perfil = [string]$ConfigRaw.perfil_predeterminado }
if ([string]::IsNullOrWhiteSpace($Perfil)) { $Perfil = "desarrollo_rtx2000ada" }

$PerfilConfig = $ConfigRaw.perfiles.$Perfil
if (-not $PerfilConfig) {
    Write-Error "ERROR: Perfil '$Perfil' no encontrado en $ArchivoConfig"
    exit 1
}

Write-Host "[1/6] Perfil seleccionado: $Perfil" -ForegroundColor Yellow
Write-Host "      Descripción: $($PerfilConfig.descripcion)" -ForegroundColor DarkGray

# 2. Prevenir Instancias Duplicadas
Write-Host "[2/6] Verificando que no existan instancias duplicadas..." -ForegroundColor Yellow

$ProcesosExistentes = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue
if ($ProcesosExistentes) {
    Write-Host "ADVERTENCIA: Ya existe un proceso 'llama-server' en ejecución." -ForegroundColor Yellow
    Write-Host "Comprobando si responde en $UrlSalud..." -ForegroundColor DarkGray
    try {
        $Resp = Invoke-RestMethod -Uri $UrlSalud -Method Get -TimeoutSec 3 -ErrorAction Stop
        if ($Resp.status -eq "ok" -or $Resp.status -eq "ready") {
            Write-Host "ÉXITO: El servidor llama-server ya está activo y respondiendo correctamente en $UrlSalud." -ForegroundColor Green
            exit 0
        }
    } catch {
        Write-Warning "El proceso llama-server existe pero no responde en $UrlSalud. Deténgalo con .\detener_llama_server.ps1 antes de reiniciar."
        exit 1
    }
}

$PuertoEnUso = Get-NetTCPConnection -LocalPort $Puerto -State Listen -ErrorAction SilentlyContinue
if ($PuertoEnUso) {
    Write-Error "ERROR: El puerto $Puerto ya está ocupado por el PID $($PuertoEnUso.OwningProcess)."
    exit 1
}

# 3. Validación y autodetección de dispositivo GPU (CUDA si hay NVIDIA, sino Vulkan).
$DispositivoGPU = if ($PerfilConfig.gpu_device) { [string]$PerfilConfig.gpu_device } else { "auto" }
$NvidiaSmiPath = if (Test-Path "C:\Windows\system32\nvidia-smi.exe") { "C:\Windows\system32\nvidia-smi.exe" } else { (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source) }
if ($DispositivoGPU -eq "auto") {
    $DispositivoGPU = if ($NvidiaSmiPath) { "CUDA0" } else { "Vulkan0" }
}
Write-Host "[3/6] Dispositivo seleccionado: $DispositivoGPU" -ForegroundColor Yellow
if ($DispositivoGPU -match '^CUDA' -and $NvidiaSmiPath) {
    try {
        $GpuRaw = (& $NvidiaSmiPath "--query-gpu=name,memory.total,memory.free" "--format=csv,noheader,nounits" 2>$null) | Out-String
        if ($GpuRaw -and $GpuRaw.Trim()) {
            $PrimeraLinea = ($GpuRaw.Trim() -split "`n")[0]
            $GpuParts = $PrimeraLinea.Split(',')
            if ($GpuParts.Count -ge 3) {
                $GpuNombre = $GpuParts[0].Trim()
                $VramTotal = [int]$GpuParts[1].Trim()
                $VramLibre = [int]$GpuParts[2].Trim()
                Write-Host "      GPU Detectada: $GpuNombre" -ForegroundColor Green
                Write-Host "      VRAM Libre: $VramLibre MB / Total: $VramTotal MB" -ForegroundColor Green

                $VramMinima = [int]$PerfilConfig.presupuesto_vram_esperado_mb
                if ($VramLibre -lt $VramMinima) {
                    Write-Warning "PRECAUCION: VRAM libre ($VramLibre MB) es menor que el presupuesto esperado ($VramMinima MB)."
                }
            }
        }
    } catch {
        Write-Warning "No se pudo consultar la VRAM con nvidia-smi: $_"
    }
} else {
    Write-Host "      Lectura nvidia-smi omitida; el dispositivo $DispositivoGPU se valida al cargar llama-server." -ForegroundColor DarkGray
}

# 4. Localizar Ejecutable llama-server.exe
Write-Host "[4/6] Verificando ejecutable de llama-server..." -ForegroundColor Yellow

$RutaBinDirecta = Join-Path $DirectorioScript "bin\llama-server.exe"
$RutaBinRaiz = Join-Path $RaizProyecto "bin\llama-server.exe"
$CmdPath = (Get-Command "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)

$RutaEjecutable = $null
if (-not [string]::IsNullOrWhiteSpace($BinarioOverride) -and (Test-Path $BinarioOverride)) {
    $RutaEjecutable = (Resolve-Path $BinarioOverride).Path
} elseif (-not [string]::IsNullOrWhiteSpace($env:LLAMA_SERVER_BIN) -and (Test-Path $env:LLAMA_SERVER_BIN)) {
    $RutaEjecutable = (Resolve-Path $env:LLAMA_SERVER_BIN).Path
} elseif (Test-Path $RutaBinDirecta) {
    $RutaEjecutable = (Resolve-Path $RutaBinDirecta).Path
} elseif (Test-Path $RutaBinRaiz) {
    $RutaEjecutable = (Resolve-Path $RutaBinRaiz).Path
} elseif ($CmdPath -and (Test-Path $CmdPath)) {
    $RutaEjecutable = (Resolve-Path $CmdPath).Path
}

if (-not $RutaEjecutable) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host " BLOQUEO DETECTADO: NO SE ENCONTRO 'llama-server.exe'" -ForegroundColor Red
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host "Para utilizar el servicio LLM en Windows con el backend del perfil '$DispositivoGPU':" -ForegroundColor Yellow
    Write-Host "1. Instale una compilacion de llama.cpp que incluya el backend de GPU seleccionado:" -ForegroundColor White
    Write-Host "   https://github.com/ggerganov/llama.cpp/releases" -ForegroundColor Cyan
    Write-Host "2. Extraiga el ejecutable en:" -ForegroundColor White
    Write-Host "   $DirectorioScript\bin\llama-server.exe" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Red
    exit 2
}

Write-Host "      Ejecutable localizado: $RutaEjecutable" -ForegroundColor Green

# 5. Localizar Archivo de Modelo GGUF (prioriza Llama 12B si existe en models/)
Write-Host "[5/6] Verificando archivo del modelo GGUF..." -ForegroundColor Yellow

$DirModelos = Join-Path $RaizProyecto "models"
$NombreModelo = [string]$PerfilConfig.modelo_archivo
$RutaModeloDefecto = if ($NombreModelo -and $NombreModelo -ne "auto") { Join-Path $RaizProyecto $PerfilConfig.modelo_ruta_relativa } else { "" }
$RutaModeloModels = if ($NombreModelo -and $NombreModelo -ne "auto") { Join-Path $DirModelos $NombreModelo } else { "" }

$RutaModelo = $null
if (-not [string]::IsNullOrWhiteSpace($ModeloOverride) -and (Test-Path $ModeloOverride -PathType Leaf)) {
    $RutaModelo = (Resolve-Path $ModeloOverride).Path
} elseif (-not [string]::IsNullOrWhiteSpace($env:LLAMA_MODEL_PATH) -and (Test-Path $env:LLAMA_MODEL_PATH -PathType Leaf)) {
    $RutaModelo = (Resolve-Path $env:LLAMA_MODEL_PATH).Path
} elseif ($RutaModeloDefecto -and (Test-Path $RutaModeloDefecto -PathType Leaf)) {
    $RutaModelo = (Resolve-Path $RutaModeloDefecto).Path
} elseif ($RutaModeloModels -and (Test-Path $RutaModeloModels -PathType Leaf)) {
    $RutaModelo = (Resolve-Path $RutaModeloModels).Path
} elseif (Test-Path $DirModelos) {
    $Ggufs = @(Get-ChildItem -Path $DirModelos -File -Filter "*.gguf" -ErrorAction SilentlyContinue)
    if ($Ggufs.Count -gt 0) {
        # Priorizar cualquier modelo 12B descargado en models/; si no hay, elegir el .gguf más grande disponible
        $Gguf12B = $Ggufs | Where-Object { $_.Name -match "12[bB]" } | Sort-Object Length -Descending | Select-Object -First 1
        $Elegido = if ($Gguf12B) { $Gguf12B } else { $Ggufs | Sort-Object Length -Descending | Select-Object -First 1 }
        $RutaModelo = $Elegido.FullName
    }
}

if (-not $RutaModelo) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host " BLOQUEO DETECTADO: MODELO GGUF NO ENCONTRADO" -ForegroundColor Red
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host "No se encontró ningún archivo .gguf en: $DirModelos" -ForegroundColor Yellow
    Write-Host "Descarga o copia tu modelo Llama 12B (formato .gguf) dentro de:" -ForegroundColor White
    Write-Host "  $DirModelos" -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Red
    exit 3
}

$ItemModelo = Get-Item $RutaModelo
$NombreModelo = $ItemModelo.Name
$TamanoGB = [math]::Round(($ItemModelo.Length / 1GB), 2)
Write-Host "      Modelo localizado: $RutaModelo ($TamanoGB GB)" -ForegroundColor Green

# 6. Lanzar Proceso
Write-Host "[6/6] Lanzando proceso llama-server en segundo plano..." -ForegroundColor Yellow

$ArgList = @(
    "-m", $RutaModelo,
    "-dev", $DispositivoGPU,
    "--host", $HostLocal,
    "--port", $Puerto,
    "-c", $PerfilConfig.n_ctx,
    "-ngl", $PerfilConfig.n_gpu_layers,
    "-b", $PerfilConfig.n_batch,
    "-t", $PerfilConfig.n_threads,
    "--parallel", $PerfilConfig.parallel
)

if ($PerfilConfig.flash_attn -eq $true) {
    $ArgList += "-fa"
    $ArgList += "on"
}

if ($PerfilConfig.kv_8bits -eq $true) {
    $ArgList += "-ctk"
    $ArgList += "q8_0"
    $ArgList += "-ctv"
    $ArgList += "q8_0"
}

# Decodificación especulativa sin un segundo modelo. El modelo principal valida
# cada token propuesto, por lo que la salida conserva el mismo resultado greedy.
$ConfigEspeculacion = $PerfilConfig.speculative_decoding
if ($ConfigEspeculacion -and $ConfigEspeculacion.enabled -eq $true) {
    $TipoEspeculacion = [string]$ConfigEspeculacion.type
    if ([string]::IsNullOrWhiteSpace($TipoEspeculacion)) {
        Write-Error "La decodificación especulativa está habilitada, pero falta speculative_decoding.type en el perfil '$Perfil'."
        exit 4
    }
    $AyudaServidor = (& $RutaEjecutable --help 2>&1 | Out-String)
    if ($AyudaServidor -match "--spec-type") {
        $ArgList += "--spec-type"
        $ArgList += $TipoEspeculacion

        $ArgumentosNgrama = [ordered]@{
            ngram_simple_size_n = "--spec-ngram-simple-size-n"
            ngram_simple_size_m = "--spec-ngram-simple-size-m"
            ngram_simple_min_hits = "--spec-ngram-simple-min-hits"
        }
        foreach ($ClaveParametro in $ArgumentosNgrama.Keys) {
            $ValorParametro = $ConfigEspeculacion.$ClaveParametro
            $ArgumentoParametro = $ArgumentosNgrama[$ClaveParametro]
            if ($null -ne $ValorParametro -and $AyudaServidor -match [regex]::Escape($ArgumentoParametro)) {
                $ArgList += $ArgumentoParametro
                $ArgList += [string]$ValorParametro
            } elseif ($null -ne $ValorParametro) {
                Write-Warning "La versión de llama-server no admite $ArgumentoParametro; se usará su valor predeterminado."
            }
        }
        Write-Host "      Decodificación especulativa: $TipoEspeculacion" -ForegroundColor Green
    } else {
        Write-Warning "La versión de llama-server no admite decodificación especulativa; se iniciará sin esa optimización."
    }
}

$Proceso = Start-Process -FilePath $RutaEjecutable `
    -ArgumentList $ArgList `
    -RedirectStandardOutput $ArchivoLog `
    -RedirectStandardError $ArchivoErr `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

if ($Proceso.HasExited) {
    if (Test-Path $ArchivoErr) { Get-Content $ArchivoErr -Tail 15 | Write-Host -ForegroundColor Red }
    if (Test-Path $ArchivoLog) { Get-Content $ArchivoLog -Tail 15 | Write-Host -ForegroundColor Yellow }
    Write-Error "ERROR: llama-server terminó inmediatamente. Revise $ArchivoLog y $ArchivoErr"
    exit 1
}

Write-Host "      Proceso iniciado con PID: $($Proceso.Id)" -ForegroundColor Green

# Registrar archivo de estado y propiedad para el supervisor y script de parada
$ArchivoEstado = Join-Path $DirectorioScript "llama_server_estado.json"
$InfoEstado = [ordered]@{
    pid = $Proceso.Id
    puerto = $Puerto
    host = $HostLocal
    perfil = $Perfil
    modelo_archivo = $NombreModelo
    modelo_ruta = $RutaModelo
    ejecutable_ruta = $RutaEjecutable
    iniciado_en = (Get-Date -Format "o")
    audacia_id = "audacia-llama-server-$Puerto"
}
$InfoEstado | ConvertTo-Json -Depth 3 | Set-Content -Path $ArchivoEstado -Encoding UTF8

# 7. Sondeo de Salud
Write-Host "Esperando inicialización y carga de capas en VRAM..." -ForegroundColor Yellow
$Listo = $false

for ($i = 1; $i -le 120; $i++) {
    Start-Sleep -Seconds 1
    if ($Proceso.HasExited) {
        if (Test-Path $ArchivoErr) { Get-Content $ArchivoErr -Tail 15 | Write-Host -ForegroundColor Red }
        if (Test-Path $ArchivoLog) { Get-Content $ArchivoLog -Tail 15 | Write-Host -ForegroundColor Yellow }
        Write-Error "ERROR: llama-server se cerró durante la carga en VRAM."
        exit 1
    }
    try {
        $RespSalud = Invoke-RestMethod -Uri $UrlSalud -Method Get -TimeoutSec 2 -ErrorAction Stop
        if ($RespSalud.status -eq "ok" -or $RespSalud.status -eq "ready") {
            $Listo = $true
            break
        }
    } catch {
        Write-Host -NoNewline "."
    }
}

Write-Host ""
if ($Listo) {
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host " SERVICIO LLM LISTO Y OPERATIVO EN http://$HostLocal`:$Puerto" -ForegroundColor Green
    Write-Host " Modelo: $NombreModelo" -ForegroundColor Green
    Write-Host " PID: $($Proceso.Id)" -ForegroundColor Green
    Write-Host "==================================================================" -ForegroundColor Green
    
    if ($Esperar) {
        Write-Host "Modo -Esperar activo: manteniendo el proceso vivo en segundo plano..." -ForegroundColor Cyan
        Wait-Process -Id $Proceso.Id
    }
} else {
    Write-Warning "El servidor no respondió en 60 segundos. Verifique $ArchivoLog."
}
