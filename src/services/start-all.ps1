# Compatibility entry point: keep one authoritative launcher at repository root.
$AgentRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
& (Join-Path $AgentRoot 'start-all.ps1') @args
