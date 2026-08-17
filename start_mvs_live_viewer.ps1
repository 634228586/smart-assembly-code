$ErrorActionPreference = 'Stop'
$PackageRoot = $PSScriptRoot
$RuntimePython = Join-Path $PackageRoot '.runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $RuntimePython -PathType Leaf)) {
    Write-Error 'Package runtime is missing. Run tools\build_runtime.ps1 first.'
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:QT_OPENGL = 'software'
$env:QT_QUICK_BACKEND = 'software'
$env:QSG_RHI_BACKEND = 'software'
Push-Location $PackageRoot
try {
    & $RuntimePython -m app.mvs_live_viewer_main
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
