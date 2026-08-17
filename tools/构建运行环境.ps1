$ErrorActionPreference = 'Stop'
$PackageRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RuntimeRoot = Join-Path $PackageRoot '.runtime'
$Requirements = Join-Path $PackageRoot 'requirements-lock.txt'
$BuildTemp = Join-Path $PackageRoot 'tmp\runtime-build'

if (Test-Path -LiteralPath $RuntimeRoot) {
    Write-Error '.runtime 已存在。为防止覆盖比赛环境，本脚本不会自动删除；请人工备份并处理。'
}

$Python = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $Python) {
    Write-Error '未找到 Windows py 启动器。需要 Python 3.11 x64。'
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
& $RuntimePython -c "import PySide6,cv2,numpy,yaml,openai,httpx,pyaubo_sdk; print('比赛运行环境导入检查通过')"

Write-Host '包内运行环境构建完成。MVS机器级驱动仍须单独安装并通过赛前检查。构建临时目录可在确认完成后删除。'
