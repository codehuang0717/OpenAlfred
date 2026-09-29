# Shared by the full-stack and screen-only launch paths.
function Stop-AlfredProcessTree([int]$ProcessId) {
    $children = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId")
    foreach ($child in $children) { Stop-AlfredProcessTree -ProcessId $child.ProcessId }
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Start-ScreenSupervisor([string]$AgentRoot, [string]$PythonExe) {
    $scriptPath = Join-Path $AgentRoot 'src\supervisor.py'
    $pattern = '(?i)(?:^|["\s])' + [regex]::Escape($scriptPath) + '(?:["\s]|$)'
    $existing = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match $pattern
    })
    $existingIds = @($existing | ForEach-Object { $_.ProcessId })
    foreach ($process in $existing) {
        if ($existingIds -notcontains $process.ParentProcessId) {
            Write-Host "[Supervisor] Restarting this repository's process tree (PID $($process.ProcessId))."
            Stop-AlfredProcessTree -ProcessId $process.ProcessId
        }
    }
    $logsDir = Join-Path $AgentRoot 'logs'
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    return Start-Process -FilePath $PythonExe -ArgumentList ('"' + $scriptPath + '"') `
        -WorkingDirectory (Join-Path $AgentRoot 'src') -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logsDir 'supervisor-stdout.log') `
        -RedirectStandardError (Join-Path $logsDir 'supervisor-stderr.log')
}
