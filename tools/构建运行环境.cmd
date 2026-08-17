@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_runtime.ps1"
exit /b %ERRORLEVEL%
