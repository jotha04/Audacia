@echo off
setlocal
title AudacIA - Hacu Visitante
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\iniciar_hacu_visitante.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Hacu Visitante finalizo con codigo %EXIT_CODE%. Revisa los mensajes anteriores.
    pause
)
exit /b %EXIT_CODE%
