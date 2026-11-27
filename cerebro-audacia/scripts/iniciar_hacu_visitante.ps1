# ==============================================================================
# AudacIA — Lanzador independiente para Hacu Visitante
# ==============================================================================
# Nota: Hacu Visitante opera de forma independiente a Hacu Inventario.
# Este script instala sus dependencias si faltan, asegura que El Cerebro (8080)
# esté activo y arranca run_hacu.py.
# ==============================================================================

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsHacu
)

$ErrorActionPreference = "Stop"

$Raiz = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DirVenvVis = Join-Path $Raiz ".venv_visitante"
$PyVenv = Join-Path $DirVenvVis "Scripts\python.exe"
$ReqBase = Join-Path $Raiz "requirements.txt"
$ReqUI = Join-Path $Raiz "requirements-ui.txt"
$ReqVoz = Join-Path $Raiz "requirements-voz.txt"
$DirModelos = Join-Path $Raiz "models"
$BinLlama = Join-Path $Raiz "cerebro_audacia\proveedor_modelo\bin\llama-server.exe"
$ScriptDescargarBin = Join-Path $Raiz "cerebro_audacia\proveedor_modelo\descargar_binarios.py"
$ScriptIniciarLLM = Join-Path $Raiz "cerebro_audacia\proveedor_modelo\iniciar_llama_server.ps1"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  AUDACIA — INICIO DE HACU VISITANTE (MODO INDEPENDIENTE)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

function Resolver-PythonBase {
    foreach ($cmd in @("python", "python3")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found) {
            try {
                $ver = & $found.Source --version 2>&1
                if ($ver -match "Python 3\.(1[0-9]|[8-9])") { return @($found.Source) }
            } catch {}
        }
    }
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher) { return @($pyLauncher.Source, "-3") }
    return $null
}

# ── 1. Crear entorno virtual dedicado para Hacu Visitante si no existe ────────
if (-not (Test-Path $PyVenv)) {
    Write-Host "[1/4] Creando entorno virtual (.venv_visitante)..." -ForegroundColor Yellow
    $PyBase = Resolver-PythonBase
    if (-not $PyBase) {
        Write-Host "[ERROR] No se encontró Python 3.10+ en el sistema." -ForegroundColor Red
        exit 1
    }
    $pyExe = $PyBase[0]
    $pyArgs = @()
    if ($PyBase.Count -gt 1) { $pyArgs += $PyBase[1..($PyBase.Count - 1)] }
    $pyArgs += @("-m", "venv", $DirVenvVis)
    & $pyExe @pyArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PyVenv)) {
        Write-Host "[ERROR] No se pudo crear .venv_visitante." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/4] Entorno .venv_visitante detectado." -ForegroundColor Green
}

# ── 2. Verificar e instalar dependencias de Hacu Visitante ────────────────────
Write-Host "[2/4] Verificando dependencias de Hacu Visitante..." -ForegroundColor Yellow
$DepsOk = $false
try {
    $resCheck = & $PyVenv -c "import chromadb, sentence_transformers, langchain_text_splitters; print('OK')" 2>$null
    if ($LASTEXITCODE -eq 0 -and "$resCheck".Trim() -eq "OK") {
        $DepsOk = $true
    }
} catch {
    $DepsOk = $false
}

if (-not $DepsOk) {
    Write-Host "      Instalando dependencias base (ChromaDB, embeddings multilingües)..." -ForegroundColor Yellow
    & $PyVenv -m pip install --upgrade pip
    # Instalar paquetes RAG sin forzar compilación C++ de llama-cpp-python (se usa llama-server HTTP)
    $TempReq = Join-Path $env:TEMP "hacu_visitante_req.txt"
    Get-Content $ReqBase | Where-Object { $_ -notmatch "^\s*llama-cpp-python" } | Set-Content -Path $TempReq -Encoding UTF8
    & $PyVenv -m pip install -r $TempReq
    Remove-Item $TempReq -Force -ErrorAction SilentlyContinue
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Falló la instalación de dependencias de Hacu Visitante." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "      Dependencias base ya instaladas." -ForegroundColor Green
}

if ($ArgsHacu -contains "--ui" -and (Test-Path $ReqUI)) {
    try { & $PyVenv -c "import PySide6" 2>$null } catch {}
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      Instalando interfaz gráfica (requirements-ui.txt)..." -ForegroundColor Yellow
        & $PyVenv -m pip install -r $ReqUI
    }
}
if (($ArgsHacu -contains "--voz" -or $ArgsHacu -contains "--voz-salida") -and (Test-Path $ReqVoz)) {
    try { & $PyVenv -c "import sounddevice, faster_whisper" 2>$null } catch {}
    if ($LASTEXITCODE -ne 0) {
        Write-Host "      Instalando módulos de voz (requirements-voz.txt)..." -ForegroundColor Yellow
        & $PyVenv -m pip install -r $ReqVoz
    }
}

# ── 3. Asegurar que llama-server (8080) esté disponible ───────────────────────
Write-Host "[3/4] Comprobando disponibilidad de llama-server (puerto 8080)..." -ForegroundColor Yellow
$LlmListo = $false
try {
    $salud = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
    if ($salud.status -eq "ok" -or $salud.status -eq "ready") {
        $LlmListo = $true
        Write-Host "      Servidor LLM activo en http://127.0.0.1:8080" -ForegroundColor Green
    }
} catch {}

if (-not $LlmListo) {
    if (-not (Test-Path $BinLlama)) {
        Write-Host "      Descargando binarios de llama-server..." -ForegroundColor Yellow
        & $PyVenv $ScriptDescargarBin
    }
    $Ggufs = @(Get-ChildItem -Path $DirModelos -File -Filter "*.gguf" -ErrorAction SilentlyContinue)
    if ($Ggufs.Count -eq 0) {
        Write-Host "[ERROR] No se encontró ningún modelo .gguf (Llama 12B) en: $DirModelos" -ForegroundColor Red
        exit 3
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptIniciarLLM
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] No se pudo iniciar llama-server." -ForegroundColor Red
        exit 4
    }
}

# ── 4. Arrancar Hacu Visitante ────────────────────────────────────────────────
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  ¡Hey! Hacu Visitante está listo — Iniciando asistente..." -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""

$env:PYTHONPATH = $Raiz
Set-Location $Raiz
if ($ArgsHacu -and $ArgsHacu.Count -gt 0) {
    & $PyVenv run_hacu.py @ArgsHacu
} else {
    & $PyVenv run_hacu.py
}
exit $LASTEXITCODE
