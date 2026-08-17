@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_tests.ps1"
exit /b %ERRORLEVEL%
