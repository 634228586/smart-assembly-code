$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$RuntimePython = Join-Path $PackageRoot '.runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $RuntimePython)) {
    Write-Error 'Package runtime is missing.'
}
$env:PYTHONUTF8 = '1'
$env:QT_QPA_PLATFORM = 'offscreen'
Push-Location $PackageRoot
try {
    & $RuntimePython -m unittest discover -s tests -v
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
