# ==============================================================================
# AudacIA — Lanzador Servidor LAN (HTTPS): El Cerebro + Hacu Inventario
# ==============================================================================
# 1. Crea .venv e instala automáticamente todos los requerimientos si faltan.
# 2. Prepara .env, migraciones, binarios llama-server y certificados TLS con la
#    IP fija del equipo (indispensable para usar el micrófono desde otros PCs/móviles).
# 3. Inicia El Cerebro (127.0.0.1:8080) y el Servidor Hacu Inventario (0.0.0.0:8000 HTTPS).
# 4. Avisa cuando todo está listo y abre el navegador automáticamente.
# ==============================================================================

param(
    [string]$IpFija = "",
    [int]$PuertoWeb = 0,
    [string]$PerfilLLM = "",
    [switch]$Http,
    [switch]$SinNavegador
)

$ErrorActionPreference = "Stop"

$Raiz = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DirInv = Join-Path $Raiz "sistema_inventario"
$DirVenv = Join-Path $Raiz ".venv"
$PyVenv = Join-Path $DirVenv "Scripts\python.exe"
$ReqDjango = Join-Path $DirInv "requirements-django.txt"
$ReqVoz = Join-Path $DirInv "requirements-haku-voz.txt"
$DirModelos = Join-Path $Raiz "models"
$BinLlama = Join-Path $Raiz "cerebro_audacia\proveedor_modelo\bin\llama-server.exe"
$ScriptDescargarBin = Join-Path $Raiz "cerebro_audacia\proveedor_modelo\descargar_binarios.py"
$ScriptIniciarLLM = Join-Path $Raiz "cerebro_audacia\proveedor_modelo\iniciar_llama_server.ps1"
$ScriptCerts = Join-Path $DirInv "scripts\plataforma\generar_certificados.ps1"
$DirCerts = Join-Path $DirInv "certs"
$CertFile = Join-Path $DirCerts "localhost.pem"
$KeyFile = Join-Path $DirCerts "localhost-key.pem"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  AUDACIA — SERVIDOR CENTRAL: EL CEREBRO + HACU INVENTARIO" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# ── Helpers ───────────────────────────────────────────────────────────────────
function Probar-PythonBase([string]$Ejecutable, [string[]]$Argumentos = @()) {
    try {
        & $Ejecutable @Argumentos -c "import struct, sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and struct.calcsize('P') == 8 else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Resolver-PythonBase {
    $pyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($pyLauncher -and (Probar-PythonBase $pyLauncher.Source @("-3.12-64"))) {
        return @($pyLauncher.Source, "-3.12-64")
    }
    foreach ($cmd in @("python3.12", "python", "python3")) {
        $found = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($found -and (Probar-PythonBase $found.Source)) {
            return @($found.Source)
        }
    }
    return $null
}

function Leer-ValorEnv([string]$Archivo, [string]$Clave) {
    if (-not (Test-Path $Archivo)) { return "" }
    foreach ($linea in (Get-Content -Path $Archivo -Encoding UTF8)) {
        if ($linea -match "^\s*$([regex]::Escape($Clave))\s*=\s*(.*)$") {
            return $Matches[1].Trim().Trim('"').Trim("'")
        }
    }
    return ""
}

function Detectar-IpLan {
    # Prioriza ruta saliente real (funciona tanto con Ethernet como con Wi-Fi de IP fija)
    try {
        $ipPy = & $PyVenv -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('8.8.8.8', 80)); print(s.getsockname()[0]); s.close()" 2>$null
        if ($ipPy -and $ipPy.Trim() -match '^\d+\.\d+\.\d+\.\d+$' -and $ipPy.Trim() -ne '127.0.0.1') {
            return $ipPy.Trim()
        }
    } catch {}
    try {
        $cand = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
            Where-Object {
                $_.InterfaceAlias -notlike "*Loopback*" -and
                $_.InterfaceAlias -notlike "*vEthernet*" -and
                $_.IPAddress -notlike "169.254*" -and
                $_.IPAddress -ne "127.0.0.1"
            } | Select-Object -ExpandProperty IPAddress -First 1
        if ($cand) { return $cand.Trim() }
    } catch {}
    return "127.0.0.1"
}

function Probar-Url([string]$Url) {
    try {
        $res = & $PyVenv -c "import ssl, urllib.request; ctx=ssl._create_unverified_context(); r=urllib.request.urlopen('$Url', context=ctx, timeout=2); print(r.status)" 2>$null
        return ("$res".Trim() -eq "200")
    } catch {
        return $false
    }
}

function Obtener-EstadoSalud([string]$Url) {
    try {
        $json = & $PyVenv -c "import ssl, urllib.request; ctx=ssl._create_unverified_context(); r=urllib.request.urlopen('$Url', context=ctx, timeout=3); print(r.read().decode('utf-8'))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $json) {
            return ($json | ConvertFrom-Json)
        }
    } catch {}
    return $null
}

# ── 1. Crear .venv e instalar requerimientos solo si faltan ───────────────────
if (-not (Test-Path $PyVenv) -or -not (Probar-PythonBase $PyVenv)) {
    Write-Host "[1/7] Creando entorno virtual (.venv)..." -ForegroundColor Yellow
    $PyBase = Resolver-PythonBase
    if (-not $PyBase) {
        Write-Host "[ERROR] Se requiere Python 3.12 de 64 bits." -ForegroundColor Red
        Write-Host "        Instálalo con 'py install 3.12' o desde https://www.python.org/downloads/." -ForegroundColor Yellow
        exit 1
    }
    if (Test-Path -LiteralPath $DirVenv) {
        $NombreRespaldo = ".venv-anterior-" + [guid]::NewGuid().ToString("N")
        Rename-Item -LiteralPath $DirVenv -NewName $NombreRespaldo
        Write-Host "      Entorno anterior conservado en $NombreRespaldo." -ForegroundColor Yellow
    }
    $pyExe = $PyBase[0]
    $pyArgs = @()
    if ($PyBase.Count -gt 1) { $pyArgs += $PyBase[1..($PyBase.Count - 1)] }
    $pyArgs += @("-m", "venv", $DirVenv)
    & $pyExe @pyArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PyVenv)) {
        Write-Host "[ERROR] No se pudo crear el entorno virtual .venv." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/7] Entorno virtual .venv detectado." -ForegroundColor Green
}

Write-Host "[2/7] Verificando dependencias de servidor y voz..." -ForegroundColor Yellow
function Probar-Dependencias([string]$Modulos) {
    try {
        & $PyVenv -c "import $Modulos" 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

$ModulosServidor = "django, uvicorn, cryptography, psycopg2, dotenv, django_extensions, openpyxl, numpy"
if (-not (Probar-Dependencias $ModulosServidor)) {
    Write-Host "      Instalando dependencias del servidor..." -ForegroundColor Yellow
    & $PyVenv -m pip install -r $ReqDjango
    if ($LASTEXITCODE -ne 0 -or -not (Probar-Dependencias $ModulosServidor)) {
        Write-Host "[ERROR] Falló la instalación de $ReqDjango." -ForegroundColor Red
        exit 1
    }
}
Write-Host "      Dependencias del servidor listas." -ForegroundColor Green

$ModulosVoz = "faster_whisper, av, piper"
if ((Test-Path $ReqVoz) -and -not (Probar-Dependencias $ModulosVoz)) {
    Write-Host "      Instalando complementos de voz..." -ForegroundColor Yellow
    & $PyVenv -m pip install -r $ReqVoz
    if ($LASTEXITCODE -ne 0 -or -not (Probar-Dependencias $ModulosVoz)) {
        Write-Warning "No se pudieron instalar todos los complementos de voz ($ReqVoz). El modo web/texto seguirá operativo."
    }
}

# ── 3. Preparar .env, IP fija del servidor y migraciones ──────────────────────
Write-Host "[3/7] Preparando configuración (.env), IP fija del servidor y base de datos..." -ForegroundColor Yellow
$EnvFile = Join-Path $DirInv ".env"
$EnvExample = Join-Path $DirInv ".env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    $secret = & $PyVenv -c "import secrets; print(secrets.token_urlsafe(50))"
    $contenidoEnv = (Get-Content -Path $EnvExample -Raw -Encoding UTF8) -replace "cambia-esto-por-una-clave-secreta-larga-y-aleatoria", $secret.Trim()
    Set-Content -Path $EnvFile -Value $contenidoEnv -Encoding UTF8
    Write-Host "      Archivo sistema_inventario\.env generado automáticamente con clave segura." -ForegroundColor Green
}

if ([string]::IsNullOrWhiteSpace($IpFija)) { $IpFija = Leer-ValorEnv $EnvFile "SERVER_IP" }
if ([string]::IsNullOrWhiteSpace($IpFija)) { $IpFija = Detectar-IpLan }
if ($PuertoWeb -le 0) {
    $pEnv = Leer-ValorEnv $EnvFile "SERVER_PORT"
    $PuertoWeb = if ($pEnv -match '^\d+$') { [int]$pEnv } else { 8000 }
}

Write-Host "      IP fija / LAN del servidor : $IpFija (puerto $PuertoWeb)" -ForegroundColor Green

$env:PYTHONPATH = "$Raiz;$DirInv"
$env:SERVER_IP = $IpFija
$env:SERVER_PORT = [string]$PuertoWeb

Push-Location $DirInv
try {
    & $PyVenv manage.py migrate --noinput | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Fallaron las migraciones de Django." -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

# ── 4. Certificados TLS para HTTPS con la IP fija del equipo ──────────────────
$UsarHttps = -not $Http
if ($UsarHttps) {
    Write-Host "[4/7] Verificando certificados TLS (HTTPS para acceso LAN y micrófono)..." -ForegroundColor Yellow
    & powershell -NoProfile -ExecutionPolicy Bypass -File $ScriptCerts -CertsDir $DirCerts -IpFija $IpFija
    if (-not (Test-Path $CertFile) -or -not (Test-Path $KeyFile)) {
        Write-Host "[ERROR] No se pudieron generar los certificados TLS en $DirCerts." -ForegroundColor Red
        exit 2
    }
    $env:SESSION_COOKIE_SECURE = "True"
    $env:CSRF_COOKIE_SECURE = "True"
} else {
    Write-Host "[4/7] Modo HTTP solicitado (-Http). Omitiendo certificados TLS." -ForegroundColor DarkGray
    $env:SESSION_COOKIE_SECURE = "False"
    $env:CSRF_COOKIE_SECURE = "False"
}

# ── 5. Verificar binario llama-server.exe y modelo GGUF (Llama 12B) ───────────
Write-Host "[5/7] Verificando motor del Cerebro Central (llama-server y modelo Llama 12B)..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path $DirModelos -Force | Out-Null
$ModelosGguf = @(Get-ChildItem -Path $DirModelos -File -Filter "*.gguf" -ErrorAction SilentlyContinue)
if ($ModelosGguf.Count -eq 0) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host " FALTA EL MODELO LLM (LLAMA 12B .gguf)" -ForegroundColor Red
    Write-Host "==================================================================" -ForegroundColor Red
    Write-Host "Coloca el archivo .gguf de Llama 12B dentro de la carpeta:" -ForegroundColor Yellow
    Write-Host "  $DirModelos" -ForegroundColor Cyan
    Write-Host "Luego vuelve a ejecutar iniciar_hacu_inventario.cmd." -ForegroundColor White
    Write-Host "==================================================================" -ForegroundColor Red
    exit 3
}

if (-not (Test-Path $BinLlama)) {
    Write-Host "      llama-server.exe no encontrado. Descargando binarios oficiales de llama.cpp..." -ForegroundColor Yellow
    & $PyVenv $ScriptDescargarBin
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $BinLlama)) {
        Write-Host "[ERROR] No se pudo descargar llama-server.exe automáticamente." -ForegroundColor Red
        exit 2
    }
}

# ── 6. Iniciar El Cerebro (llama-server interno en 127.0.0.1:8080) ────────────
Write-Host "[6/7] Iniciando El Cerebro Central (puerto 8080)..." -ForegroundColor Yellow
$CerebroListo = $false
try {
    $salud = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -Method Get -TimeoutSec 2 -ErrorAction Stop
    if ($salud.status -eq "ok" -or $salud.status -eq "ready") {
        $CerebroListo = $true
        Write-Host "      El Cerebro ya estaba en línea en http://127.0.0.1:8080" -ForegroundColor Green
    }
} catch {}

if (-not $CerebroListo) {
    $argsLlm = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptIniciarLLM)
    if (-not [string]::IsNullOrWhiteSpace($PerfilLLM)) {
        $argsLlm += @("-Perfil", $PerfilLLM)
    }
    & powershell @argsLlm
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] No se pudo iniciar El Cerebro (llama-server)." -ForegroundColor Red
        exit 4
    }
}

# ── 7. Iniciar Servidor Hacu Inventario (0.0.0.0:PuertoWeb, Uvicorn HTTPS) ───
$Esquema = if ($UsarHttps) { "https" } else { "http" }
$UrlLocal = "${Esquema}://localhost:${PuertoWeb}/"
$UrlLan = "${Esquema}://${IpFija}:${PuertoWeb}/"
$UrlCheck = "${Esquema}://127.0.0.1:${PuertoWeb}/cuentas/login/"
$UrlSalud = "${Esquema}://127.0.0.1:${PuertoWeb}/api/haku/salud/"

Write-Host "[7/7] Iniciando servidor Hacu Inventario en 0.0.0.0:${PuertoWeb} ($($Esquema.ToUpper()))..." -ForegroundColor Yellow

$WebYaCorriendo = Probar-Url $UrlCheck
$ProcWeb = $null
if (-not $WebYaCorriendo) {
    # Liberar puerto si quedó un proceso huérfano previo
    $Occupied = Get-NetTCPConnection -LocalPort $PuertoWeb -State Listen -ErrorAction SilentlyContinue
    if ($Occupied) {
        foreach ($conn in $Occupied) {
            if ($conn.OwningProcess -gt 4) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
        Start-Sleep -Milliseconds 600
    }

    if ($UsarHttps) {
        $ArgsWeb = @(
            "-m", "uvicorn", "inventario_audacia.asgi:application",
            "--host", "0.0.0.0",
            "--port", [string]$PuertoWeb,
            "--ssl-certfile", $CertFile,
            "--ssl-keyfile", $KeyFile
        )
    } else {
        $ArgsWeb = @(
            "-m", "uvicorn", "inventario_audacia.asgi:application",
            "--host", "0.0.0.0",
            "--port", [string]$PuertoWeb
        )
    }

    $ProcWeb = Start-Process -FilePath $PyVenv `
        -ArgumentList $ArgsWeb `
        -WorkingDirectory $DirInv `
        -NoNewWindow `
        -PassThru

    $WebLista = $false
    for ($i = 1; $i -le 60; $i++) {
        Start-Sleep -Seconds 1
        if ($ProcWeb.HasExited) {
            Write-Host "[ERROR] El servidor Uvicorn/Django se detuvo inesperadamente." -ForegroundColor Red
            exit 5
        }
        if (Probar-Url $UrlCheck) {
            $WebLista = $true
            break
        }
    }

    if (-not $WebLista) {
        Write-Host "[ERROR] El servidor Hacu Inventario tardó demasiado en responder en $UrlLocal." -ForegroundColor Red
        exit 5
    }
}

# Esperar el tiempo necesario hasta que la precarga de modelos de voz (STT/VAD/TTS) termine por completo
$SaludInicial = Obtener-EstadoSalud $UrlSalud
if ($SaludInicial -and $SaludInicial.warmup_iniciado -and -not $SaludInicial.warmup_completado) {
    Write-Host "      Esperando carga completa de modelos de voz (STT / VAD / TTS)..." -ForegroundColor Yellow
    Write-Host "      (La primera vez puede tardar un par de minutos mientras descarga e inicializa Whisper)." -ForegroundColor DarkGray
    $InicioEspera = Get-Date
    $UltimoLog = -10
    while ($true) {
        if ($ProcWeb -and $ProcWeb.HasExited) {
            Write-Host "[ERROR] El servidor Uvicorn/Django se detuvo mientras cargaba los modelos de voz." -ForegroundColor Red
            exit 5
        }
        Start-Sleep -Seconds 2
        $SaludActual = Obtener-EstadoSalud $UrlSalud
        if ($SaludActual -and $SaludActual.warmup_completado) {
            $SaludInicial = $SaludActual
            break
        }
        $Transcurrido = [int]((Get-Date) - $InicioEspera).TotalSeconds
        if (($Transcurrido - $UltimoLog) -ge 8) {
            $UltimoLog = $Transcurrido
            $EstadoVad = if ($SaludActual -and $SaludActual.vad_listo) { "OK" } else { "cargando" }
            $EstadoStt = if ($SaludActual -and $SaludActual.stt_listo) { "OK" } else { "cargando" }
            $EstadoTts = if ($SaludActual -and $SaludActual.tts_listo) { "OK" } else { "pendiente" }
            Write-Host ("      ... inicializando motores de voz ({0}s) [VAD: {1} | STT: {2} | TTS: {3}]" -f $Transcurrido, $EstadoVad, $EstadoStt, $EstadoTts) -ForegroundColor DarkYellow
        }
    }
}

if ($SaludInicial) {
    if ($SaludInicial.stt_listo) {
        $DispStt = if ($SaludInicial.stt_dispositivo) { $SaludInicial.stt_dispositivo } else { "activo" }
        Write-Host "      Modelos de voz inicializados correctamente (STT: $DispStt)." -ForegroundColor Green
    } else {
        Write-Warning "El servidor web inició, pero el motor de voz STT reportó advertencias durante la precarga. Revisa el log anterior."
    }
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  ¡Hey! Ya está listo — El Cerebro y Servidor Hacu Inventario" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  Acceso en red (IP fija) : $UrlLan" -ForegroundColor White
Write-Host "  Panel de Voz (LAN)      : ${UrlLan}api/haku/panel/" -ForegroundColor White
Write-Host "  Acceso en este servidor : $UrlLocal" -ForegroundColor White
Write-Host "  Cerebro Central (LLM)   : http://127.0.0.1:8080/health" -ForegroundColor White
Write-Host "  ----------------------------------------------------------------" -ForegroundColor DarkGray
Write-Host "  * Otros PCs/celulares en la red entran por: $UrlLan" -ForegroundColor DarkGray
Write-Host "  * Si el Firewall bloquea otros equipos, ejecuta como Admin:" -ForegroundColor DarkGray
Write-Host "    netsh advfirewall firewall add rule name=`"Hacu Inventario`" dir=in action=allow protocol=TCP localport=$PuertoWeb" -ForegroundColor DarkGray
Write-Host "  * Presiona Ctrl + C en esta ventana para detener el servidor." -ForegroundColor DarkGray
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""

if (-not $SinNavegador) {
    $UrlAbrir = if ($IpFija -and $IpFija -ne "127.0.0.1") { $UrlLan } else { $UrlLocal }
    Start-Process $UrlAbrir
}

if ($ProcWeb) {
    try {
        Wait-Process -Id $ProcWeb.Id
    } finally {
        if (-not $ProcWeb.HasExited) {
            Stop-Process -Id $ProcWeb.Id -Force -ErrorAction SilentlyContinue
        }
    }
}
