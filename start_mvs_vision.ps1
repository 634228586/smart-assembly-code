$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RuntimePython = Join-Path $PackageRoot '.runtime\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
    Write-Error 'Package runtime is missing. Run tools\build_runtime.ps1 first.'
}

$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

Push-Location $PackageRoot
try {
    & $RuntimePython -m vision.real_mvs_service
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
