@echo off
rem Pemasang Prakiraan Saham Bank BEI untuk Windows 10/11 64-bit, tanpa hak admin.
rem Cukup klik dua kali file ini; tidak ada yang perlu diketik. Instalasi butuh internet,
rem lalu aplikasi terbuka sendiri di browser. Rincian ada di installer\windows\pasang.ps1.
rem Bila folder installer tidak ada di samping file ini (misalnya file ini dijalankan langsung
rem dari dalam ZIP), kode aplikasi diunduh dulu dari GitHub.
setlocal
title Pemasang Prakiraan Saham Bank BEI
set "SCRIPT=%~dp0installer\windows\pasang.ps1"
set "SRC="
if exist "%SCRIPT%" goto install

echo Mengunduh kode aplikasi dari GitHub ...
set "SRC=%TEMP%\PrakiraanSahamBEI-unduhan"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12; $src = $env:SRC; if (Test-Path $src) { Remove-Item $src -Recurse -Force }; New-Item -ItemType Directory -Path $src | Out-Null; $zip = Join-Path $src 'aplikasi.zip'; Invoke-WebRequest -UseBasicParsing -Uri 'https://github.com/ahsan4th/idx-bank-forecast/archive/refs/heads/main.zip' -OutFile $zip; Expand-Archive -Path $zip -DestinationPath $src -Force"
if errorlevel 1 goto failed
set "SCRIPT=%SRC%\idx-bank-forecast-main\installer\windows\pasang.ps1"
if not exist "%SCRIPT%" goto failed

:install
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%"
if errorlevel 1 goto failed
if defined SRC rmdir /s /q "%SRC%"
echo.
echo Jendela ini menutup sendiri dalam 15 detik.
timeout /t 15 >nul
exit /b 0

:failed
echo.
echo Instalasi belum berhasil. Baca pesan di atas, lalu klik lagi pasang_windows.bat.
pause
exit /b 1
