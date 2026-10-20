@echo off
setlocal
cd /d "%~dp0.."
where powershell.exe >nul 2>&1 || (echo PowerShell introuvable.& exit /b 1)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_menu.ps1" %*
exit /b %ERRORLEVEL%
