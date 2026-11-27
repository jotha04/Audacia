@echo off
setlocal
title AudacIA - Cerebro Central + Hacu Inventario
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\iniciar_hacu_inventario.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] El arranque finalizo con codigo %EXIT_CODE%. Revisa los mensajes anteriores.
    pause
)
exit /b %EXIT_CODE%
