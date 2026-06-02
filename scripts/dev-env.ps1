param(
    [switch]$DownloadOrt = $true,
    [switch]$RunCheck = $true,
    [switch]$RunBuild = $true,
    [switch]$RunSmoke = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ortVersion = "1.24.4"
$ortDir = Join-Path $repoRoot "vendor\onnxruntime"
$ortArchive = Join-Path $ortDir "onnxruntime-win-x64-$ortVersion.zip"
$ortExtractRoot = Join-Path $ortDir "onnxruntime-win-x64-$ortVersion"
$ortDll = Join-Path $ortExtractRoot "onnxruntime-win-x64-$ortVersion\lib\onnxruntime.dll"

if ($DownloadOrt -and -not (Test-Path $ortDll)) {
    New-Item -ItemType Directory -Force -Path $ortDir | Out-Null

    if (-not (Test-Path $ortArchive)) {
        $url = "https://github.com/microsoft/onnxruntime/releases/download/v$ortVersion/onnxruntime-win-x64-$ortVersion.zip"
        Write-Host "Downloading ONNX Runtime from $url"
        curl.exe -L -o $ortArchive $url
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to download ONNX Runtime archive."
        }
    }

    if (Test-Path $ortExtractRoot) {
        Remove-Item -Recurse -Force $ortExtractRoot
    }
    Expand-Archive -Path $ortArchive -DestinationPath $ortExtractRoot
}

if (-not (Test-Path $ortDll)) {
    throw "ONNX Runtime DLL not found at: $ortDll"
}

$env:ORT_DYLIB_PATH = $ortDll
Write-Host "ORT_DYLIB_PATH=$env:ORT_DYLIB_PATH"

Push-Location $repoRoot
try {
    if ($RunCheck) {
        cargo check
        if ($LASTEXITCODE -ne 0) {
            throw "cargo check failed."
        }
    }

    if ($RunBuild) {
        cargo build
        if ($LASTEXITCODE -ne 0) {
            throw "cargo build failed."
        }
    }

    if ($RunSmoke) {
        $extensionCandidates = @()
        $extensionCandidates += Get-ChildItem -Path (Join-Path $repoRoot "target\debug\*deepface*.dll") -File -ErrorAction SilentlyContinue
        $extensionCandidates += Get-ChildItem -Path (Join-Path $repoRoot "target\debug\deps\*deepface*.dll") -File -ErrorAction SilentlyContinue
        $extensionDll = $extensionCandidates | Select-Object -First 1
        if (-not $extensionDll) {
            throw "Extension DLL not found in target\debug or target\debug\deps."
        }

        php -n -d "extension=$($extensionDll.FullName)" "scripts/smoke_extension.php"
        if ($LASTEXITCODE -ne 0) {
            throw "smoke_extension.php failed."
        }
    }
}
finally {
    Pop-Location
}
