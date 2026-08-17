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
foreach ($VariableName in @(
    'AUBO_RPC_PASSWORD',
    'DASHSCOPE_API_KEY',
    'DASHSCOPE_BASE_URL',
    'DASHSCOPE_MODEL'
)) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($VariableName, 'Process'))) {
        $UserValue = [Environment]::GetEnvironmentVariable($VariableName, 'User')
        if (-not [string]::IsNullOrWhiteSpace($UserValue)) {
            [Environment]::SetEnvironmentVariable($VariableName, $UserValue, 'Process')
        }
    }
}
Push-Location $PackageRoot
try {
    & $RuntimePython -m app.main
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
