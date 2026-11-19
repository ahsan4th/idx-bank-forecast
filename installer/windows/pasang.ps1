# Pemasang "Prakiraan Saham Bank BEI" untuk Windows 10/11 64-bit, per pengguna, tanpa hak admin.
#
# Jalankan lewat pasang_windows.bat di folder utama aplikasi. Instalasi butuh internet.
# Setelah itu aplikasi berjalan di PC ini dan bobot model tidak diunduh lagi; harga saham
# dan berita tetap diambil online saat aplikasi dipakai.
#
# Semua yang dipasang ada di %LOCALAPPDATA%\PrakiraanSahamBEI:
#   app\      salinan kode aplikasi
#   uv\       uv (pengelola Python dan paket dari Astral), versi dipin, checksum diperiksa
#   python\   Python 3.12 milik uv
#   venv\     lingkungan Python dengan paket dari requirements.txt
#   hf-home\  bobot model foundation (cache Hugging Face)
# ditambah shortcut di Desktop dan Start Menu serta entri uninstall di Settings > Apps.
# Menjalankan ulang pemasang memperbarui instalasi yang sudah ada.

#Requires -Version 5.1
Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # the progress bar makes Invoke-WebRequest very slow on 5.1
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$AppName       = 'Prakiraan Saham Bank BEI'
$AppId         = 'PrakiraanSahamBEI'
$Publisher     = 'Muhammad Ahsan, ITS'
$UvVersion     = '0.12.5'
$UvAsset       = 'uv-x86_64-pc-windows-msvc.zip'
$PythonRequest = 'cpython-3.12-windows-x86_64-none'
$PythonMinor   = '3.12'
$MinFreeGB     = 6

$Source      = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$InstallDir  = Join-Path $env:LOCALAPPDATA $AppId
$AppDir      = Join-Path $InstallDir 'app'
$UvDir       = Join-Path $InstallDir 'uv'
$UvExe       = Join-Path $UvDir 'uv.exe'
$VenvDir     = Join-Path $InstallDir 'venv'
$VenvPython  = Join-Path $VenvDir 'Scripts\python.exe'
$Launcher    = Join-Path $InstallDir 'jalankan.bat'
$Uninstaller = Join-Path $InstallDir 'hapus.bat'
$Scripts     = @('jalankan.bat', 'hapus.bat', 'hapus.ps1')
$LogFile     = Join-Path $InstallDir 'pasang.log'

function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Note([string]$Text) { Write-Host "    $Text" }
function Write-Warn([string]$Text) { Write-Host "    PERINGATAN: $Text" -ForegroundColor Yellow }

function Invoke-Native([string]$Exe, [string[]]$Arguments) {
    # Runs a program, shows its output indented, and returns its exit code. These tools
    # write progress to stderr, so stderr must not stop the script; the exit code decides.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "      $_" }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    return $code
}

function Assert-Architecture {
    $arch = [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITEW6432')
    if (-not $arch) { $arch = [Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITECTURE') }
    if ($arch -eq 'AMD64') { return }
    if ($arch -eq 'ARM64') {
        Write-Warn 'Windows ARM terdeteksi. Versi x64 dipasang dan berjalan lewat emulasi Windows; kombinasi ini belum diuji.'
        return
    }
    throw "Arsitektur $arch tidak didukung: aplikasi ini butuh Windows 64-bit."
}

function Assert-FreeSpace {
    $drive = (Split-Path -Qualifier $env:LOCALAPPDATA).TrimEnd(':')
    $freeGB = [math]::Round((Get-PSDrive -Name $drive).Free / 1GB, 1)
    Write-Note "Ruang kosong di drive ${drive}: $freeGB GB"
    if ($freeGB -ge $MinFreeGB) { return }
    if (Test-Path $VenvPython) {
        Write-Warn "Ruang kosong kurang dari $MinFreeGB GB, tetapi instalasi lama dipakai ulang, jadi pembaruan tetap dicoba."
        return
    }
    throw "Ruang kosong di drive ${drive} hanya $freeGB GB, padahal instalasi butuh sekitar $MinFreeGB GB. Kosongkan ruang, lalu klik lagi pasang_windows.bat."
}

function Copy-AppFiles {
    if ([string]::Equals($Source.TrimEnd('\'), $AppDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
        Write-Note 'Pemasang dijalankan dari folder yang sudah terpasang; file aplikasi tidak disalin ulang.'
        return
    }
    $settings = Join-Path $AppDir 'settings.toml'
    if (Test-Path $settings) {
        $backup = Join-Path $InstallDir ('settings.toml.bak-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
        Copy-Item -Path $settings -Destination $backup
        Write-Note "settings.toml sebelumnya disimpan sebagai $backup"
    }
    if (Test-Path $AppDir) { Remove-Item -Path $AppDir -Recurse -Force }
    $code = Invoke-Native 'robocopy.exe' @($Source, $AppDir, '/E', '/XD', '.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '/XF', '*.pyc', '/NFL', '/NDL', '/NJH', '/NJS', '/NP')
    if ($code -ge 8) { throw "Menyalin file aplikasi gagal (robocopy kode $code)." }
    Write-Note "Kode aplikasi disalin ke $AppDir"
}

function Install-Uv {
    if (Test-Path $UvExe) {
        $current = & $UvExe --version | Select-Object -First 1
        if ($current -like "uv $UvVersion*") { Write-Note "uv $UvVersion sudah ada."; return }
        Remove-Item -Path $UvDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
    $zip = Join-Path $env:TEMP "$AppId-$UvAsset"
    $bases = @(
        "https://github.com/astral-sh/uv/releases/download/$UvVersion",
        "https://releases.astral.sh/github/uv/releases/download/$UvVersion"
    )
    $downloaded = $false
    foreach ($base in $bases) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$UvAsset" -OutFile $zip
            Invoke-WebRequest -UseBasicParsing -Uri "$base/$UvAsset.sha256" -OutFile "$zip.sha256"
            $downloaded = $true
            break
        } catch {
            Write-Warn "Unduhan dari $base gagal: $($_.Exception.Message)"
        }
    }
    if (-not $downloaded) { throw 'uv tidak bisa diunduh. Periksa koneksi internet, lalu jalankan ulang pemasang.' }
    $expected = ((Get-Content -Path "$zip.sha256" -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zip).Hash.ToLowerInvariant()
    if ($expected -ne $actual) {
        Remove-Item -Path $zip, "$zip.sha256" -Force
        throw "Checksum uv tidak cocok (seharusnya $expected, didapat $actual). Jalankan ulang pemasang."
    }
    Expand-Archive -Path $zip -DestinationPath $UvDir -Force
    Remove-Item -Path $zip, "$zip.sha256" -Force
    if (-not (Test-Path $UvExe)) {
        $found = Get-ChildItem -Path $UvDir -Recurse -Filter 'uv.exe' | Select-Object -First 1
        if (-not $found) { throw 'uv.exe tidak ditemukan di arsip uv.' }
        Move-Item -Path $found.FullName -Destination $UvExe -Force
    }
    Write-Note "uv $UvVersion dipasang (checksum SHA-256 cocok)."
}

function Install-Python {
    $code = Invoke-Native $UvExe @('python', 'install', $PythonRequest)
    if ($code -ne 0) { throw "Instalasi Python $PythonMinor gagal." }
    $found = & $UvExe python find $PythonRequest | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or -not $found) { throw "Python $PythonMinor tidak ditemukan setelah dipasang." }
    return $found.Trim()
}

function Test-Venv {
    if (-not (Test-Path $VenvPython)) { return $false }
    $version = & $VenvPython -c "import sys; print('%d.%d' % sys.version_info[:2])"
    return ($LASTEXITCODE -eq 0 -and $version -eq $PythonMinor)
}

function New-Venv([string]$BasePython) {
    if (Test-Venv) { Write-Note 'Lingkungan Python yang sudah ada dipakai ulang.'; return }
    if (Test-Path $VenvDir) { Remove-Item -Path $VenvDir -Recurse -Force }
    $code = Invoke-Native $UvExe @('venv', $VenvDir, '--python', $BasePython)
    if ($code -ne 0) { throw 'Membuat lingkungan Python gagal.' }
}

function Install-Packages {
    # unsafe-best-match: every index is considered and the highest version wins, so torch
    # comes as the CPU build and all other packages come from PyPI.
    $full = Join-Path $AppDir 'requirements.txt'
    $code = Invoke-Native $UvExe @('pip', 'install', '--python', $VenvPython, '--index-strategy', 'unsafe-best-match', '-r', $full)
    if ($code -eq 0) { return 'lengkap' }
    # One click must still end in a working app: install the light set instead, and the
    # final summary (and the app's model list) says that the foundation models are missing.
    Write-Warn 'Paket lengkap (torch dan model foundation) gagal dipasang; lihat pesan di atas. Versi ringan dipasang sebagai gantinya.'
    $code = Invoke-Native $UvExe @('pip', 'install', '--python', $VenvPython, '-r', (Join-Path $AppDir 'requirements-lite.txt'))
    if ($code -ne 0) { throw 'Paket Python gagal dipasang, juga versi ringannya. Periksa koneksi internet, lalu klik lagi pasang_windows.bat.' }
    return 'ringan'
}

function Save-PackageList {
    $list = & $UvExe pip freeze --python $VenvPython
    if ($LASTEXITCODE -ne 0) { throw 'Daftar paket terpasang tidak bisa dibaca.' }
    $path = Join-Path $InstallDir 'paket-terpasang.txt'
    $list | Set-Content -Path $path -Encoding ASCII
    Write-Note "Versi paket dicatat di $path"
}

function Get-ModelWeights {
    $code = Invoke-Native $VenvPython @((Join-Path $AppDir 'installer\unduh_model.py'))
    if ($code -eq 1) {
        Write-Warn 'Sebagian bobot model gagal diunduh. Model itu tampil sebagai error di aplikasi; jalankan ulang pemasang untuk mencoba lagi.'
    } elseif ($code -ne 0) {
        throw "Pengunduh model berhenti dengan kode $code."
    }
}

function Test-Installation {
    $env:HF_HUB_OFFLINE = '1'
    try {
        $code = Invoke-Native $VenvPython @((Join-Path $AppDir 'installer\cek_instalasi.py'))
    } finally {
        Remove-Item -Path Env:\HF_HUB_OFFLINE -ErrorAction SilentlyContinue
    }
    if ($code -eq 3) {
        Write-Warn 'Sebagian model foundation belum bisa dimuat (lihat daftar di atas); aplikasi akan melewatinya.'
    } elseif ($code -ne 0) {
        throw "Pemeriksaan instalasi gagal (kode $code); lihat pesan di atas."
    }
}

function Install-Scripts {
    foreach ($name in $Scripts) {
        Copy-Item -Path (Join-Path $AppDir "installer\windows\$name") -Destination (Join-Path $InstallDir $name) -Force
    }
    # Streamlit asks for an e-mail address on its very first run; an empty one skips the question.
    $streamlitDir = Join-Path $env:USERPROFILE '.streamlit'
    $credentials = Join-Path $streamlitDir 'credentials.toml'
    if (-not (Test-Path $credentials)) {
        New-Item -ItemType Directory -Force -Path $streamlitDir | Out-Null
        Set-Content -Path $credentials -Value "[general]`r`nemail = `"`"" -Encoding ASCII
    }
}

function Unblock-InstalledFiles {
    # Files from a downloaded ZIP carry the internet "mark of the web". Removing it keeps
    # Windows from showing a security warning every time the shortcut starts the app.
    Get-ChildItem -Path $AppDir -Recurse -File | Unblock-File
    foreach ($name in $Scripts) { Unblock-File -Path (Join-Path $InstallDir $name) }
}

function New-Shortcut([string]$Path, [string]$Target, [string]$WorkDir, [string]$Description, [string]$Icon) {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut($Path)
    $link.TargetPath = $Target
    $link.WorkingDirectory = $WorkDir
    $link.Description = $Description
    if ($Icon) { $link.IconLocation = $Icon }
    $link.Save()
}

function Install-Shortcuts([string]$Icon) {
    $description = 'Buka aplikasi prakiraan harga saham bank BEI'
    $desktop = [Environment]::GetFolderPath('Desktop')
    $menu = Join-Path ([Environment]::GetFolderPath('Programs')) $AppName
    New-Item -ItemType Directory -Force -Path $menu | Out-Null
    New-Shortcut (Join-Path $desktop "$AppName.lnk") $Launcher $AppDir $description $Icon
    New-Shortcut (Join-Path $menu "$AppName.lnk") $Launcher $AppDir $description $Icon
    New-Shortcut (Join-Path $menu "Hapus $AppName.lnk") $Uninstaller $env:TEMP "Hapus $AppName dari PC ini" ''
    Write-Note "Shortcut dibuat di Desktop dan di Start Menu > $AppName"
}

function Register-Uninstall([string]$Icon) {
    $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppId"
    New-Item -Path $key -Force | Out-Null
    $bytes = (Get-ChildItem -Path $InstallDir -Recurse -File -Force | Measure-Object -Property Length -Sum).Sum
    $strings = [ordered]@{
        DisplayName     = $AppName
        DisplayVersion  = (Get-Date -Format 'yyyy.MM.dd')
        Publisher       = $Publisher
        InstallLocation = $InstallDir
        DisplayIcon     = $Icon
        UninstallString = "`"$Uninstaller`""
    }
    foreach ($name in $strings.Keys) {
        New-ItemProperty -Path $key -Name $name -Value $strings[$name] -PropertyType String -Force | Out-Null
    }
    $numbers = [ordered]@{ NoModify = 1; NoRepair = 1; EstimatedSize = [int]($bytes / 1KB) }
    foreach ($name in $numbers.Keys) {
        New-ItemProperty -Path $key -Name $name -Value $numbers[$name] -PropertyType DWord -Force | Out-Null
    }
    Write-Note 'Terdaftar di Settings > Apps, jadi bisa dihapus dari sana.'
}

function Clear-UvCache {
    $code = Invoke-Native $UvExe @('cache', 'clean')
    if ($code -ne 0) { Write-Warn 'Cache unduhan uv tidak bisa dibersihkan; aplikasi tetap bisa dipakai.' }
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Start-Transcript -Path $LogFile -Append | Out-Null
$exitCode = 0
try {
    Write-Host "Pemasang $AppName" -ForegroundColor Cyan
    Write-Note "Sumber : $Source"
    Write-Note "Tujuan : $InstallDir"
    Write-Note 'Instalasi butuh internet dan bisa cukup lama, tergantung kecepatan koneksi.'
    Write-Note 'Tidak ada yang perlu diketik: tunggu sampai aplikasi terbuka sendiri di browser.'

    Write-Step 'Memeriksa PC'
    Assert-Architecture
    Assert-FreeSpace

    # Keep Python, the download cache and the model weights inside the install folder,
    # and ignore any user-level uv configuration so every PC installs the same way. The
    # cache sits on the same drive as the venv, so uv hardlinks files instead of copying.
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallDir 'python'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:UV_CACHE_DIR = Join-Path $InstallDir 'uv-cache'
    $env:UV_NO_CONFIG = '1'
    $env:PYTHONUTF8 = '1'
    $env:HF_HOME = Join-Path $InstallDir 'hf-home'
    $env:HF_HUB_DISABLE_TELEMETRY = '1'
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

    Write-Step '1/7 Menyalin file aplikasi'
    Copy-AppFiles
    Write-Step "2/7 Memasang uv $UvVersion"
    Install-Uv
    Write-Step "3/7 Memasang Python $PythonMinor"
    $basePython = Install-Python
    New-Venv $basePython
    Write-Step '4/7 Memasang paket Python (torch, model foundation, dan lainnya)'
    $mode = Install-Packages
    Save-PackageList
    Write-Step '5/7 Mengunduh bobot model foundation'
    Get-ModelWeights
    Write-Step '6/7 Memeriksa instalasi'
    Test-Installation
    Write-Step '7/7 Membuat shortcut'
    Install-Scripts
    Unblock-InstalledFiles
    Install-Shortcuts "$basePython,0"
    Register-Uninstall "$basePython,0"
    Clear-UvCache

    Write-Step 'Selesai'
    Write-Note "$AppName terpasang di $InstallDir."
    if ($mode -eq 'ringan') {
        Write-Warn 'Yang terpasang versi ringan (random walk dan gradient boosting). Klik lagi pasang_windows.bat untuk mencoba versi lengkap.'
    }
    Write-Note 'Aplikasi sedang dibuka di browser; harga saham dan berita tetap butuh internet.'
    Write-Note "Lain kali, buka lewat shortcut '$AppName' di Desktop atau Start Menu."
    Start-Process -FilePath $Launcher -WorkingDirectory $AppDir
} catch {
    Write-Host ''
    Write-Host "GAGAL: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Catatan lengkap ada di $LogFile"
    $exitCode = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $exitCode
