<#
.SYNOPSIS
    OpenAlfred - Start All Services
.DESCRIPTION
    Starts all OpenAlfred services locally (no Docker required for core):
      1. Redis                     (port 6379, uses Memurai on Windows)
      2. LiveKit Server            (port 7880, local binary)
      3. LangGraph Dev Server      (port 2024)
      4. FastAPI Business API      (port 7788)
      5. Background Worker         (reminder scheduler)
      6. Next.js Frontend          (port 3000)
      7. LiveKit Cloud Worker      (voice calls, port 5883)
      8. LiveKit Local Worker      (wake-word voice, port 5884)
      9. Ear Service               (microphone wake-word)
     10. Supervisor                (proactive screen monitoring)

    Press Ctrl+C to stop all services.
#>

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Write-Service($name, $color, $msg) {
    Write-Host "[$name]" -ForegroundColor $color -NoNewline
    Write-Host " $msg"
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   OpenAlfred - Starting All Services    " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

$AgentDir   = Join-Path $Root "agent"
$WebDir     = Join-Path $Root "web"
$SrcDir     = Join-Path $AgentDir "src"
$VenvPython = Join-Path $AgentDir ".venv\Scripts\python.exe"
$VenvLangGraph = Join-Path $AgentDir ".venv\Scripts\langgraph.exe"
$LiveKitServer = Join-Path $Root "bin\livekit-server.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: Python venv not found at $VenvPython" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VenvLangGraph)) {
    Write-Host "ERROR: LangGraph CLI not found at $VenvLangGraph. Run 'uv sync' in agent dir." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $LiveKitServer)) {
    Write-Host "ERROR: LiveKit Server not found at $LiveKitServer." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $WebDir "node_modules"))) {
    Write-Host "ERROR: web/node_modules not found. Run 'cd web; npm install' first." -ForegroundColor Red
    exit 1
}

# Find Redis (Memurai on Windows, redis-server elsewhere)
$redisExe = $null
if (Get-Command redis-server -ErrorAction SilentlyContinue) {
    $redisExe = (Get-Command redis-server).Source
} elseif (Get-Command memurai -ErrorAction SilentlyContinue) {
    $redisExe = (Get-Command memurai).Source
} elseif (Get-Command memurai-server -ErrorAction SilentlyContinue) {
    $redisExe = (Get-Command memurai-server).Source
} elseif (Test-Path "C:\Program Files\Redis\redis-server.exe") {
    $redisExe = "C:\Program Files\Redis\redis-server.exe"
} elseif (Test-Path "C:\Program Files\Memurai\memurai.exe") {
    $redisExe = "C:\Program Files\Memurai\memurai.exe"
}

if (-not $redisExe) {
    Write-Host "ERROR: Redis not found (checked redis-server, memurai, memurai-server)." -ForegroundColor Red
    Write-Host "  Install Memurai on Windows: winget install Memurai" -ForegroundColor Red
    Write-Host "  Or install Redis on Linux:   apt install redis" -ForegroundColor Red
    exit 1
}

# Helper: check if a TCP port is already listening.
# Test-NetConnection is painfully slow on Windows when the port is closed, so use TcpClient.
function Test-PortInUse($port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect("127.0.0.1", $port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(500, $false)) {
            return $false
        }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Test-HttpOk($url) {
    try {
        $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        return ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500)
    } catch {
        return $false
    }
}

$env:PYTHONPATH = $SrcDir

$processes = @()

# 0. Redis
$redisAlreadyRunning = Test-PortInUse 6379
if ($redisAlreadyRunning) {
    Write-Service "Redis" "DarkYellow" "Port 6379 already in use — using existing Redis instance."
    $processes += $null  # placeholder to keep indices aligned
} elseif ($redisExe) {
    Write-Service "Redis" "DarkYellow" "Starting Redis on port 6379..."
    $redisWorkDir = Split-Path -Parent $redisExe
    $processes += Start-Process -FilePath $redisExe `
        -WorkingDirectory $redisWorkDir `
        -PassThru -WindowStyle Hidden
} else {
    Write-Host "ERROR: Redis not found and port 6379 is not active." -ForegroundColor Red
    exit 1
}

# 1. LiveKit Local Server
$lkAlreadyRunning = Test-PortInUse 7880
if ($lkAlreadyRunning) {
    Write-Service "LiveKit-Server" "Cyan" "Port 7880 already in use — using existing LiveKit instance."
    $processes += $null
} else {
    Write-Service "LiveKit-Server" "Cyan" "Starting local LiveKit server..."
    $logsDir = Join-Path $Root "logs"
    if (-not (Test-Path $logsDir)) { New-Item -ItemType Directory -Path $logsDir -Force | Out-Null }
    $processes += Start-Process -FilePath $LiveKitServer `
        -ArgumentList "--dev", "--bind", "127.0.0.1" `
        -WorkingDirectory $Root `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardError (Join-Path $logsDir "livekit-stderr.log") `
        -RedirectStandardOutput (Join-Path $logsDir "livekit-stdout.log")
}

# 2. LangGraph Dev Server
$lgAlreadyRunning = Test-PortInUse 2024
if ($lgAlreadyRunning) {
    Write-Service "LangGraph" "Magenta" "Port 2024 already in use — using existing LangGraph instance."
    $processes += $null
} else {
    Write-Service "LangGraph" "Magenta" "Starting on port 2024..."
    $env:LANGGRAPH_NO_VERSION_CHECK = "true"
    $processes += Start-Process -FilePath $VenvLangGraph `
        -ArgumentList "dev", "--allow-blocking", "--no-browser" `
        -WorkingDirectory $AgentDir `
        -PassThru -NoNewWindow
}

# 3. FastAPI Business API
$apiAlreadyRunning = Test-PortInUse 7788
if ($apiAlreadyRunning) {
    Write-Service "FastAPI" "Green" "Port 7788 already in use — using existing FastAPI instance."
    $processes += $null
} else {
    Write-Service "FastAPI" "Green" "Starting on port 7788..."
    $processes += Start-Process -FilePath $VenvPython `
        -ArgumentList "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7788" `
        -WorkingDirectory $SrcDir `
        -PassThru -NoNewWindow
}

# --- Backend Synchronization ---
function Wait-For-Backend($port, $name, $healthUrl = "") {
    Write-Host "[$name]" -ForegroundColor Gray -NoNewline
    Write-Host " Waiting for port $port..." -NoNewline
    $maxWait = 120
    for ($i = 0; $i -lt $maxWait; $i++) {
        if ($healthUrl) {
            $check = Test-HttpOk $healthUrl
        } else {
            $check = Test-PortInUse $port
        }
        if ($check) {
            Write-Host " UP!" -ForegroundColor Green
            return $true
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 1
    }
    Write-Host " TIMEOUT! Continuing anyway..." -ForegroundColor Yellow
    return $false
}

Write-Host ""
Wait-For-Backend 6379 "Redis" | Out-Null
Wait-For-Backend 7880 "LiveKit-Server" | Out-Null
Wait-For-Backend 2024 "LangGraph" "http://127.0.0.1:2024/ok" | Out-Null
Wait-For-Backend 7788 "FastAPI" "http://127.0.0.1:7788/" | Out-Null
Write-Host "Backend services are ready. Starting dependent services..." -ForegroundColor White
Write-Host ""

# 3. Background Worker
Write-Service "Worker" "Yellow" "Starting background worker..."
$processes += Start-Process -FilePath $VenvPython `
    -ArgumentList (Join-Path $SrcDir "worker.py") `
    -WorkingDirectory $SrcDir `
    -PassThru -NoNewWindow

# 4. Next.js Frontend
$frontendAlreadyRunning = Test-PortInUse 3000
if ($frontendAlreadyRunning) {
    Write-Service "Frontend" "DarkYellow" "Port 3000 already in use — using existing frontend instance."
    $processes += $null
} else {
    Write-Service "Frontend" "Blue" "Starting on port 3000..."
    $npmCmd = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $npmCmd) { $npmCmd = (Get-Command npm -ErrorAction Stop).Source }
    $processes += Start-Process -FilePath $npmCmd `
        -ArgumentList "run", "dev" `
        -WorkingDirectory $WebDir `
        -PassThru -NoNewWindow
}

# 5. LiveKit Cloud Worker (For Phone Calls)
Write-Service "LK-Cloud" "DarkCyan" "Starting Cloud Voice Worker..."
$lkCloudArgs = (Join-Path $SrcDir "livekit_worker.py") + " dev"
$OriginalURL = $env:LIVEKIT_URL
$env:LIVEKIT_HTTP_SERVER_PORT = "5883"
$processes += Start-Process -FilePath $VenvPython `
    -ArgumentList $lkCloudArgs `
    -WorkingDirectory $SrcDir `
    -PassThru -NoNewWindow

# 6. LiveKit Local Worker (For Wake-Word)
Write-Service "LK-Local" "Cyan" "Starting Local Voice Worker..."
$lkLocalArgs = (Join-Path $SrcDir "livekit_worker.py") + " dev --local"
$env:LIVEKIT_HTTP_SERVER_PORT = "5884"
$processes += Start-Process -FilePath $VenvPython `
    -ArgumentList $lkLocalArgs `
    -WorkingDirectory $SrcDir `
    -PassThru -NoNewWindow

# Restore for other services
$env:LIVEKIT_URL = $OriginalURL

# 7. Ear Service (Wake-Word Listener)
Write-Service "Ear" "Green" "Starting Wake-Word Listener..."
$processes += Start-Process -FilePath $VenvPython `
    -ArgumentList "-m", "body.windows_system.ear.ear_service", "--local" `
    -WorkingDirectory $SrcDir `
    -PassThru -NoNewWindow

# 8. The Supervisor
Write-Service "Alfred" "Red" "Starting Proactive Supervisor..."
$processes += Start-Process -FilePath $VenvPython `
    -ArgumentList (Join-Path $SrcDir "supervisor.py") `
    -WorkingDirectory $SrcDir `
    -PassThru -NoNewWindow

$serviceNames = @("Redis", "LiveKit-Server", "LangGraph", "FastAPI", "Worker", "Frontend", "LK-Cloud", "LK-Local", "Ear", "Supervisor")

# Services that should be auto-restarted on crash
$autoRestartable = @{
    "LiveKit-Server" = @{ Args = @("--dev", "--bind", "127.0.0.1"); Exe = $LiveKitServer; Dir = $Root }
    "Worker"     = @{ Args = @((Join-Path $SrcDir "worker.py")); Exe = $VenvPython; Dir = $SrcDir }
    "LK-Cloud"   = @{ Args = @(((Join-Path $SrcDir "livekit_worker.py") + " dev")); Exe = $VenvPython; Dir = $SrcDir; Env = @{ "LIVEKIT_HTTP_SERVER_PORT" = "5883" } }
    "LK-Local"   = @{ Args = @(((Join-Path $SrcDir "livekit_worker.py") + " dev --local")); Exe = $VenvPython; Dir = $SrcDir; Env = @{ "LIVEKIT_HTTP_SERVER_PORT" = "5884" } }
    "Ear"        = @{ Args = @("-m", "body.windows_system.ear.ear_service", "--local"); Exe = $VenvPython; Dir = $SrcDir }
    "Supervisor" = @{ Args = @((Join-Path $SrcDir "supervisor.py")); Exe = $VenvPython; Dir = $SrcDir }
}
$maxRestarts = 3
$restartCounts = @{}
foreach ($name in $serviceNames) { $restartCounts[$name] = 0 }
$reportedExits = @{}

Write-Host ""
Write-Host "All 10 services started! Alfred is now watching over you." -ForegroundColor Green
Write-Host ""
Write-Host "  Frontend:   http://localhost:3000" -ForegroundColor Cyan
Write-Host "  FastAPI:    http://localhost:7788" -ForegroundColor Cyan
Write-Host "  LangGraph:  http://localhost:2024" -ForegroundColor Cyan
Write-Host "  Redis:      localhost:6379" -ForegroundColor Cyan
Write-Host "  LiveKit:    Local Server (7880) & Voice Workers" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop all services..." -ForegroundColor DarkGray
Write-Host ""

try {
    while ($true) {
        for ($i = 0; $i -lt $processes.Count; $i++) {
            # Skip placeholder entries for already-running services
            if ($null -eq $processes[$i]) { continue }

            $name = $serviceNames[$i]
            if ($processes[$i].HasExited) {
                $exitCode = $processes[$i].ExitCode
                $ts = Get-Date -Format "HH:mm:ss"

                if ($autoRestartable.ContainsKey($name) -and $restartCounts[$name] -lt $maxRestarts) {
                    $restartCounts[$name]++
                    $attempt = $restartCounts[$name]
                    Write-Host "[$ts] WARNING: $name crashed (exit $exitCode). Waiting 2s then restarting ($attempt/$maxRestarts)..." -ForegroundColor Yellow

                    Start-Sleep -Seconds 2

                    $cfg = $autoRestartable[$name]

                    # Set environment variables if defined
                    if ($cfg.Env) {
                        foreach ($key in $cfg.Env.Keys) {
                            Set-Item "env:$key" $cfg.Env[$key]
                        }
                    }

                    $processes[$i] = Start-Process -FilePath $cfg.Exe `
                        -ArgumentList $cfg.Args `
                        -WorkingDirectory $cfg.Dir `
                        -PassThru -NoNewWindow

                    Write-Host "[$ts] $name restarted (PID $($processes[$i].Id))" -ForegroundColor Green
                }
                elseif ($autoRestartable.ContainsKey($name) -and $restartCounts[$name] -ge $maxRestarts) {
                    # Only warn once after max restarts
                    if ($restartCounts[$name] -eq $maxRestarts) {
                        Write-Host "[$ts] CRITICAL: $name has crashed $maxRestarts times. Giving up." -ForegroundColor Red
                        $restartCounts[$name]++  # Increment past max to suppress future warnings
                    }
                }
                else {
                    if (-not $reportedExits.ContainsKey($name)) {
                        Write-Host "[$ts] WARNING: $name exited (code $exitCode). Manual restart required." -ForegroundColor Red
                        $reportedExits[$name] = $true
                    }
                }
            }
        }
        Start-Sleep -Seconds 5
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping all services..." -ForegroundColor Yellow
    foreach ($proc in $processes) {
        if ($null -eq $proc) { continue }
        if (-not $proc.HasExited) {
            try {
                Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $proc.Id } | ForEach-Object {
                    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
                }
                Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            }
            catch { }
        }
    }
    Write-Host "All services stopped." -ForegroundColor Green
}
