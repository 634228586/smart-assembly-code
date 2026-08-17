@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_competition.ps1"
exit /b %ERRORLEVEL%
