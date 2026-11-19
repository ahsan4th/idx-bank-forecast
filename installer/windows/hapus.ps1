# Menghapus "Prakiraan Saham Bank BEI" dari PC ini: folder instalasi, shortcut, dan entri
# uninstall. Dijalankan lewat hapus.bat, shortcut di Start Menu, atau Settings > Apps.
# Konfirmasi memakai kotak dialog Ya/Tidak, jadi tidak ada yang perlu diketik.

#Requires -Version 5.1
Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$AppName    = 'Prakiraan Saham Bank BEI'
$AppId      = 'PrakiraanSahamBEI'
$InstallDir = $PSScriptRoot
$Expected   = Join-Path $env:LOCALAPPDATA $AppId
$Title      = "Hapus $AppName"

# WScript.Shell.Popup: 4 = Yes/No buttons, 32 = question icon, 64 = information, 16 = error;
# it returns 6 when Yes is clicked.
$shell = New-Object -ComObject WScript.Shell

$exitCode = 0
try {
    # Refuse to delete anything but the folder the installer created.
    if (-not [string]::Equals($InstallDir.TrimEnd('\'), $Expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Skrip ini hanya menghapus $Expected, bukan $InstallDir."
    }
    $question = "Hapus $AppName dari PC ini?`n`nFolder $InstallDir ikut terhapus, termasuk Python, paket, bobot model, dan cadangan settings.toml."
    if ($shell.Popup($question, 0, $Title, 4 + 32) -ne 6) { exit 0 }

    Set-Location -Path $env:TEMP
    $running = @(Get-CimInstance -ClassName Win32_Process | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallDir, [StringComparison]::OrdinalIgnoreCase)
    })
    foreach ($process in $running) {
        Write-Host "Menghentikan $($process.Name) (PID $($process.ProcessId)) yang masih berjalan."
        Stop-Process -Id $process.ProcessId -Force
    }

    $desktopLink = Join-Path ([Environment]::GetFolderPath('Desktop')) "$AppName.lnk"
    $menu = Join-Path ([Environment]::GetFolderPath('Programs')) $AppName
    foreach ($path in @($desktopLink, $menu)) {
        if (Test-Path $path) { Remove-Item -Path $path -Recurse -Force }
    }
    $key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\$AppId"
    if (Test-Path $key) { Remove-Item -Path $key -Recurse -Force }

    # This script lives inside the folder, so a separate process deletes it a moment later.
    $command = "/c timeout /t 3 /nobreak >nul & rmdir /s /q `"$InstallDir`""
    Start-Process -FilePath (Join-Path $env:SystemRoot 'System32\cmd.exe') -ArgumentList $command -WindowStyle Hidden
    $shell.Popup("$AppName sudah dihapus dari PC ini.", 0, $Title, 64) | Out-Null
} catch {
    $shell.Popup("Penghapusan gagal: $($_.Exception.Message)", 0, $Title, 16) | Out-Null
    $exitCode = 1
}
exit $exitCode
