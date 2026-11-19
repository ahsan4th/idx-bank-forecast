@echo off
rem Menghapus Prakiraan Saham Bank BEI dari PC ini (memanggil hapus.ps1).
rem Folder kerja dipindah ke TEMP agar folder instalasi tidak terkunci saat dihapus, dan
rem perintah exit ada di baris yang sama supaya cmd tidak membaca file ini lagi setelah terhapus.
cd /d "%TEMP%"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0hapus.ps1" & exit /b
