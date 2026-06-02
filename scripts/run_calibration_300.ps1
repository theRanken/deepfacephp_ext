param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [string]$PhpExe = "C:\laragon\bin\php\php-8.3.8-Win32-vs16-x64\php.exe",
    [string]$ExtensionPath = ".\target\debug\deps\php_deepface.dll",
    [int]$Workers = 4,
    [int]$TotalPairs = 300,
    [double]$NegativeRatio = 0.5,
    [int]$MaxPositivePerId = 10,
    [double]$Step = 0.01,
    [int]$Seed = 42,
    [string]$OutputDir = ".\tmp\calibration-300"
)

$ErrorActionPreference = "Stop"

function Fail([string]$Message) {
    Write-Error "CALIBRATION300_FAIL: $Message"
    exit 1
}

if (-not (Test-Path $PhpExe)) {
    Fail "php.exe not found at: $PhpExe"
}
if (-not (Test-Path $ExtensionPath)) {
    Fail "Extension DLL not found at: $ExtensionPath"
}
if (-not (Test-Path $Dataset)) {
    Fail "Dataset path not found: $Dataset"
}
if (-not (Test-Path $Model)) {
    Fail "Model path not found: $Model"
}
if ($Workers -lt 1) {
    Fail "Workers must be >= 1"
}
if ($TotalPairs -lt $Workers) {
    Fail "TotalPairs must be >= Workers"
}

if (-not (Test-Path $OutputDir)) {
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
}

# Required runtime env for compare calls.
# Respect pre-set env vars; only apply defaults when missing.
if ([string]::IsNullOrWhiteSpace($env:ORT_DYLIB_PATH)) {
    $env:ORT_DYLIB_PATH = "C:/Users/ranke/deepfacephp_ext/vendor/onnxruntime/onnxruntime-win-x64-1.24.4/onnxruntime-win-x64-1.24.4/lib/onnxruntime.dll"
}
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_DETECTOR_MODEL_PATH)) {
    $env:DEEPFACE_DETECTOR_MODEL_PATH = "C:/Users/ranke/deepfacephp_ext/models/scrfd_10g_gnkps.onnx"
}
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_DETECT_CONFIDENCE)) { $env:DEEPFACE_DETECT_CONFIDENCE = "0.6" }
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_MIN_FACE_SIZE)) { $env:DEEPFACE_MIN_FACE_SIZE = "40" }
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_MIN_SHARPNESS)) { $env:DEEPFACE_MIN_SHARPNESS = "8" }
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_DETECT_NMS_IOU)) { $env:DEEPFACE_DETECT_NMS_IOU = "0.45" }
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_PAIR_MARGIN)) { $env:DEEPFACE_PAIR_MARGIN = "0.02" }
if ([string]::IsNullOrWhiteSpace($env:DEEPFACE_DIAGNOSTICS)) { $env:DEEPFACE_DIAGNOSTICS = "0" }

$basePerWorker = [math]::Floor($TotalPairs / $Workers)
$remainder = $TotalPairs % $Workers

$procs = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $workerPairs = $basePerWorker + ($(if ($i -lt $remainder) { 1 } else { 0 }))
    $outLogPath = Join-Path $OutputDir ("worker-{0}.out.log" -f $i)
    $errLogPath = Join-Path $OutputDir ("worker-{0}.err.log" -f $i)
    $scorePath = Join-Path $OutputDir ("scores-w{0}.json" -f $i)
    $cachePath = Join-Path $OutputDir ("cache-w{0}.json" -f $i)

    $args = @(
        "-n",
        "-d", "extension=$ExtensionPath",
        "scripts/calibrate_threshold.php",
        "--dataset=$Dataset",
        "--model=$Model",
        "--max-positive-per-id=$MaxPositivePerId",
        "--negative-ratio=$NegativeRatio",
        "--max-total-pairs=$workerPairs",
        "--workers=$Workers",
        "--worker-index=$i",
        "--cache-file=$cachePath",
        "--dump-scores=$scorePath",
        "--progress-every=25",
        "--step=$Step",
        "--seed=$Seed"
    )

    Write-Host "Starting worker $i (pairs=$workerPairs) -> $outLogPath / $errLogPath"
    $p = Start-Process -FilePath $PhpExe -ArgumentList $args -RedirectStandardOutput $outLogPath -RedirectStandardError $errLogPath -PassThru -NoNewWindow
    $procs += $p
}

Write-Host "Waiting for workers..."
Wait-Process -Id ($procs | Select-Object -ExpandProperty Id)

$failed = $false
for ($i = 0; $i -lt $Workers; $i++) {
    $outLogPath = Join-Path $OutputDir ("worker-{0}.out.log" -f $i)
    $errLogPath = Join-Path $OutputDir ("worker-{0}.err.log" -f $i)
    $scorePath = Join-Path $OutputDir ("scores-w{0}.json" -f $i)
    if (-not (Test-Path $scorePath)) {
        Write-Host "Worker $i missing score dump: $scorePath"
        if (Test-Path $outLogPath) {
            Write-Host "--- worker $i stdout tail ---"
            Get-Content $outLogPath -Tail 40
        }
        if (Test-Path $errLogPath) {
            Write-Host "--- worker $i stderr tail ---"
            Get-Content $errLogPath -Tail 40
        }
        $failed = $true
    }
}
if ($failed) {
    Fail "One or more workers failed; see logs in $OutputDir"
}

$scoreInputs = @()
for ($i = 0; $i -lt $Workers; $i++) {
    $scoreInputs += (Join-Path $OutputDir ("scores-w{0}.json" -f $i))
}
$inputsArg = ($scoreInputs -join ",")

Write-Host "Merging worker outputs..."
& $PhpExe "scripts/merge_calibration_scores.php" "--inputs" $inputsArg "--step" "$Step"

Write-Host "CALIBRATION300_OK: output_dir=$OutputDir"
