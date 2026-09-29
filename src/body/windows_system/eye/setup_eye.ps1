param([switch]$CheckOnly, [switch]$InstallModels)
$ErrorActionPreference = "Stop"
$AgentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
Push-Location $AgentRoot
try {
    if ($InstallModels) {
        $env:PYTHONPATH = Join-Path $AgentRoot "src"
        uv run python -c "from services.screenpipe_models import install_models; print('Verified models installed:', install_models())"
    } elseif ($CheckOnly) {
        # Version and model-cache checks only: never captures or migrates recordings.
        $env:PYTHONPATH = Join-Path $AgentRoot "src"
        uv run python -c "import subprocess; from core.config import config; from services.screenpipe_models import verify_models; subprocess.run([config.SCREENPIPE_EXE, '--version'], check=True); verify_models(); print('Screenpipe models verified')"
    } else {
        # Same owner binding and explicit opt-in as start-all.
        & (Join-Path $AgentRoot 'start-all.ps1') -SupervisorOnly
    }
    if ($LASTEXITCODE -ne 0) { throw "Screen monitor command failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
