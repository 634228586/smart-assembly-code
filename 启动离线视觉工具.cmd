@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_offline_vision.ps1"
exit /b %ERRORLEVEL%
