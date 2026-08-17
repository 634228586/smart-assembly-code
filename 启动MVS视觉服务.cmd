@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_mvs_vision.ps1"
if errorlevel 1 pause
exit /b %ERRORLEVEL%
