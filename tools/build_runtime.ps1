$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent $PSScriptRoot
$RuntimeRoot = Join-Path $PackageRoot '.runtime'
$Requirements = Join-Path $PackageRoot 'requirements-lock.txt'
$BuildTemp = Join-Path $PackageRoot 'tmp\runtime-build'
if (Test-Path -LiteralPath $RuntimeRoot) {
    Write-Error '.runtime already exists; it will not be overwritten.'
}
if ($null -eq (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error 'Windows Python launcher was not found. Python 3.11 x64 is required.'
}
& py -3.11 -c "import struct,sys; assert sys.version_info[:2] == (3,11); assert struct.calcsize('P') == 8"
& py -3.11 -m venv --copies $RuntimeRoot
$RuntimePython = Join-Path $RuntimeRoot 'Scripts\python.exe'
$null = New-Item -ItemType Directory -Force -Path $BuildTemp
$env:TEMP = $BuildTemp
$env:TMP = $BuildTemp
$env:PIP_CACHE_DIR = Join-Path $BuildTemp 'pip-cache'
& $RuntimePython -m pip install --upgrade pip
& $RuntimePython -m pip install -r $Requirements
& $RuntimePython -m pip check
& $RuntimePython -c "import PySide6,cv2,numpy,yaml,openai,httpx,pyaubo_sdk; print('runtime import check passed')"
Write-Host 'Package runtime completed. Machine-level MVS drivers are still required.'
