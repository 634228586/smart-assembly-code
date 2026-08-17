@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_mvs_live_viewer.ps1"
exit /b %ERRORLEVEL%
