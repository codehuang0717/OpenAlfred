$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExePath = Join-Path $Root "screenpipe-0.3.6-x86_64-pc-windows-msvc\bin\screenpipe.exe"

function Setup-Eye {
    if (-not (Test-Path $ExePath)) {
        Write-Host "ERROR: screenpipe.exe not found at $ExePath" -ForegroundColor Red
        return $false
    }

    Write-Host "[Eye] Unblocking binary..." -ForegroundColor Gray
    Unblock-File -Path $ExePath

    # Check for the speaker model to prevent the Protobuf crash
    $ModelPath = "$env:LOCALAPPDATA\screenpipe\models\wespeaker_en_voxceleb_CAM++.onnx"
    if (-not (Test-Path $ModelPath)) {
        Write-Host "[Eye] WARNING: Speaker model missing. Screenpipe might crash or attempt a slow download." -ForegroundColor Yellow
    }

    return $true
}

function Start-Eye {
    $DataDir = Join-Path $Root "data"
    if (-not (Test-Path $DataDir)) {
        New-Item -Path $DataDir -ItemType Directory | Out-Null
        Write-Host "[Eye] Created local data directory at $DataDir" -ForegroundColor Gray
    }

    Write-Host "[Eye] Starting Screenpipe (Audio: Disabled, Data: Local)..." -ForegroundColor Cyan
    
    # We use --disable-audio as requested. 
    # --data-dir points to the local folder.
    $process = Start-Process -FilePath $ExePath `
        -ArgumentList "--disable-audio", "--fps", "1", "--data-dir", $DataDir, "--disable-telemetry" `
        -WorkingDirectory (Split-Path $ExePath) `
        -PassThru -NoNewWindow
    
    return $process
}

if ($MyInvocation.InvocationName -ne '.') {
    if (Setup-Eye) {
        $p = Start-Eye
        Write-Host "[Eye] Running with PID $($p.Id). Press Ctrl+C to stop (if running standalone)."
        try { $p | Wait-Process } finally { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
}
