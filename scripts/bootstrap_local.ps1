[CmdletBinding()]
param(
    [switch]$RecreateVenv,
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @()
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Arguments -join ' ')"
    }
}

function Install-Requirements {
    param(
        [string]$PythonExePath,
        [string]$RequirementsPath
    )
    Write-Step "Installing requirements"
    & $PythonExePath -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -eq 0) {
        return
    }

    $vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) {
        throw "pip install failed and vcvars64.bat was not found. Install Visual Studio Build Tools."
    }

    Write-Step "Retrying requirements install with Visual Studio Build Tools environment"
    $cmd = "`"$vcvars`" && `"$PythonExePath`" -m pip install -r `"$RequirementsPath`""
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed even with vcvars64.bat."
    }
}

function Set-DotEnvValue {
    param(
        [string]$EnvPath,
        [string]$Key,
        [string]$Value
    )
    $lines = @()
    if (Test-Path -LiteralPath $EnvPath) {
        $lines = Get-Content -LiteralPath $EnvPath
    }
    $needle = "^$([regex]::Escape($Key))="
    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $needle) {
            $lines[$i] = "$Key=$Value"
            $updated = $true
            break
        }
    }
    if (-not $updated) {
        $lines += "$Key=$Value"
    }
    Set-Content -LiteralPath $EnvPath -Value $lines -Encoding UTF8
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $ProjectRoot

$PythonVersion = "3.12.10"
$PythonZipName = "python-$PythonVersion-amd64.zip"
$PythonZipUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonZipName"
$PythonRoot = Join-Path $ProjectRoot "tools\\python312"
$PythonRuntime = Join-Path $PythonRoot "runtime"
$PythonZipPath = Join-Path $PythonRoot $PythonZipName
$PythonExe = Join-Path $PythonRuntime "python.exe"

Write-Step "Preparing local Python $PythonVersion"
Ensure-Dir -Path $PythonRoot
if ($ForceDownload -or -not (Test-Path -LiteralPath $PythonZipPath)) {
    Write-Step "Downloading $PythonZipName"
    Invoke-WebRequest -Uri $PythonZipUrl -OutFile $PythonZipPath
}
if ($ForceDownload -or -not (Test-Path -LiteralPath $PythonExe)) {
    if (Test-Path -LiteralPath $PythonRuntime) {
        Remove-Item -LiteralPath $PythonRuntime -Recurse -Force
    }
    Write-Step "Extracting Python runtime"
    Expand-Archive -LiteralPath $PythonZipPath -DestinationPath $PythonRuntime
}
Invoke-Checked -FilePath $PythonExe -Arguments @("--version")

$VenvPython = Join-Path $ProjectRoot ".venv\\Scripts\\python.exe"
if ($RecreateVenv -and (Test-Path -LiteralPath ".venv")) {
    Write-Step "Recreating .venv"
    Remove-Item -LiteralPath ".venv" -Recurse -Force
}
if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Step "Creating project .venv"
    & $PythonExe -m venv ".venv"
}

Write-Step "Installing Python packages into .venv"
Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Install-Requirements -PythonExePath $VenvPython -RequirementsPath "requirements.txt"

$UnidicDicDir = Join-Path $ProjectRoot ".venv\\Lib\\site-packages\\unidic\\dicdir"
if (-not (Test-Path -LiteralPath $UnidicDicDir)) {
    Write-Step "Downloading UniDic dictionary (required for Japanese)"
    try {
        Invoke-Checked -FilePath $VenvPython -Arguments @("-m", "unidic", "download")
    }
    catch {
        Write-Warning "UniDic download failed: $($_.Exception.Message)"
        Write-Warning "Japanese voice support may fail until UniDic is downloaded."
    }
}

$FfmpegVersion = "8.0.1"
$FfmpegZipName = "ffmpeg-$FfmpegVersion-essentials_build.zip"
$FfmpegUrl = "https://github.com/GyanD/codexffmpeg/releases/download/$FfmpegVersion/$FfmpegZipName"
$FfmpegRoot = Join-Path $ProjectRoot "tools\\ffmpeg"
$FfmpegZip = Join-Path $FfmpegRoot $FfmpegZipName

Write-Step "Preparing local ffmpeg"
Ensure-Dir -Path $FfmpegRoot
if ($ForceDownload -or -not (Test-Path -LiteralPath $FfmpegZip)) {
    Write-Step "Downloading ffmpeg"
    Invoke-WebRequest -Uri $FfmpegUrl -OutFile $FfmpegZip
}
if ($ForceDownload -or -not (Get-ChildItem -LiteralPath $FfmpegRoot -Recurse -Filter "ffmpeg.exe" -ErrorAction SilentlyContinue)) {
    Get-ChildItem -LiteralPath $FfmpegRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "ffmpeg-*" } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $FfmpegZip -DestinationPath $FfmpegRoot
}

$ffmpegExe = Get-ChildItem -LiteralPath $FfmpegRoot -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1 -ExpandProperty FullName
$ffprobeExe = Get-ChildItem -LiteralPath $FfmpegRoot -Recurse -Filter "ffprobe.exe" | Select-Object -First 1 -ExpandProperty FullName

$EspeakVersion = "1.52.0"
$EspeakMsiName = "espeak-ng.msi"
$EspeakUrl = "https://github.com/espeak-ng/espeak-ng/releases/download/$EspeakVersion/$EspeakMsiName"
$EspeakRoot = Join-Path $ProjectRoot "tools\\espeak"
$EspeakMsi = Join-Path $EspeakRoot $EspeakMsiName
$EspeakRuntime = Join-Path $EspeakRoot "runtime"

Write-Step "Preparing local espeak-ng"
Ensure-Dir -Path $EspeakRoot
if ($ForceDownload -or -not (Test-Path -LiteralPath $EspeakMsi)) {
    Write-Step "Downloading espeak-ng MSI"
    Invoke-WebRequest -Uri $EspeakUrl -OutFile $EspeakMsi
}
if ($ForceDownload -or -not (Get-ChildItem -LiteralPath $EspeakRoot -Recurse -Filter "libespeak-ng.dll" -ErrorAction SilentlyContinue)) {
    if (Test-Path -LiteralPath $EspeakRuntime) {
        Remove-Item -LiteralPath $EspeakRuntime -Recurse -Force
    }
    Ensure-Dir -Path $EspeakRuntime
    $msiArgs = @("/a", $EspeakMsi, "/qn", "TARGETDIR=$EspeakRuntime")
    $msiProc = Start-Process -FilePath "msiexec.exe" -ArgumentList $msiArgs -PassThru -Wait
    if ($msiProc.ExitCode -ne 0) {
        throw "msiexec extraction failed with exit code $($msiProc.ExitCode)"
    }
}

$espeakDataPath = Get-ChildItem -LiteralPath $EspeakRoot -Recurse -Directory |
    Where-Object { $_.Name -ieq "espeak-ng-data" } |
    Select-Object -First 1 -ExpandProperty FullName
$espeakDll = Get-ChildItem -LiteralPath $EspeakRoot -Recurse -Filter "libespeak-ng.dll" |
    Select-Object -First 1 -ExpandProperty FullName

if (-not (Test-Path -LiteralPath ".env")) {
    Write-Step "Creating .env from .env.example"
    if (Test-Path -LiteralPath ".env.example") {
        Copy-Item -LiteralPath ".env.example" -Destination ".env"
    }
    else {
        Set-Content -LiteralPath ".env" -Value @() -Encoding UTF8
    }
}

Write-Step "Updating .env with discovered local tool paths"
if ($ffmpegExe) { Set-DotEnvValue -EnvPath ".env" -Key "FFMPEG_BINARY" -Value $ffmpegExe }
if ($ffprobeExe) { Set-DotEnvValue -EnvPath ".env" -Key "FFPROBE_BINARY" -Value $ffprobeExe }
if ($espeakDataPath) { Set-DotEnvValue -EnvPath ".env" -Key "ESPEAK_DATA_PATH" -Value $espeakDataPath }
if ($espeakDll) { Set-DotEnvValue -EnvPath ".env" -Key "PHONEMIZER_ESPEAK_LIBRARY" -Value $espeakDll }

Write-Step "Bootstrap complete"
Write-Host "Use '.\\run.bat' to launch the app." -ForegroundColor Green
Write-Host "Use 'powershell -ExecutionPolicy Bypass -File scripts\\doctor.ps1' for diagnostics." -ForegroundColor Green
