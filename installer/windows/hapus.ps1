# Menghapus "Prakiraan Saham Bank BEI" dari PC ini: folder instalasi, shortcut, dan entri
# uninstall. Dijalankan lewat hapus.bat, shortcut di Start Menu, atau Settings > Apps.

#Requires -Version 5.1
Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$AppName    = 'Prakiraan Saham Bank BEI'
$AppId      = 'PrakiraanSahamBEI'
$InstallDir = $PSScriptRoot
$Expected   = Join-Path $env:LOCALAPPDATA $AppId

$exitCode = 0
try {
    # Refuse to delete anything but the folder the installer created.
    if (-not [string]::Equals($InstallDir.TrimEnd('\'), $Expected, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Skrip ini hanya menghapus $Expected, bukan $InstallDir."
    }
    Write-Host "Ini akan menghapus $AppName dari $InstallDir,"
    Write-Host 'termasuk Python, paket, bobot model, dan cadangan settings.toml di folder itu.'
    $answer = Read-Host 'Lanjutkan? (Y/N)'
    if ($answer -notmatch '^\s*[Yy]') {
        Write-Host 'Dibatalkan.'
    } else {
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
        Write-Host "$AppName dihapus. Foldernya hilang dalam beberapa detik."
    }
} catch {
    Write-Host "GAGAL: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
}
Read-Host 'Tekan Enter untuk menutup jendela ini' | Out-Null
exit $exitCode
