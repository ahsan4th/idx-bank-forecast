@echo off
rem Menjalankan Prakiraan Saham Bank BEI. Aplikasi terbuka di browser;
rem tutup jendela ini untuk menghentikannya. Dibuat oleh pasang_windows.bat.
setlocal
set "BASE=%~dp0"
set "PY=%BASE%venv\Scripts\python.exe"
if not exist "%PY%" goto missing

rem Bobot model dibaca dari cache lokal saja; tidak ada unduhan model saat aplikasi jalan.
set "HF_HOME=%BASE%hf-home"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONUTF8=1"

title Prakiraan Saham Bank BEI
rem Streamlit membaca .streamlit\config.toml (tema) dari folder kerja.
cd /d "%BASE%app"
echo Prakiraan Saham Bank BEI sedang dibuka di browser.
echo Biarkan jendela ini terbuka selama aplikasi dipakai. Tutup jendela ini untuk berhenti.
echo.
"%PY%" -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.headless false --browser.gatherUsageStats false
if errorlevel 1 pause
exit /b

:missing
echo Instalasi tidak lengkap: %PY% tidak ditemukan.
echo Jalankan ulang pasang_windows.bat dari folder aplikasi.
pause
exit /b 1
