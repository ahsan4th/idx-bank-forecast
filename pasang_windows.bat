@echo off
rem Pemasang Prakiraan Saham Bank BEI untuk Windows 10/11 64-bit, tanpa hak admin.
rem Klik dua kali file ini. Instalasi butuh internet; setelah itu aplikasi berjalan di PC ini.
rem Rincian ada di installer\windows\pasang.ps1 dan di README (bagian Instal di PC Windows).
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer\windows\pasang.ps1"
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
