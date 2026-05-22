<?php
declare(strict_types=1);

function fail(string $message, int $code = 1): never
{
    fwrite(STDERR, "SMOKE_FAIL: {$message}\n");
    exit($code);
}

if (!function_exists('deepface_analyze') || !function_exists('deepface_compare')) {
    fail('Extension functions are not available. Ensure the extension DLL is loaded.');
}

$ortDylibPath = getenv('ORT_DYLIB_PATH');
if ($ortDylibPath === false || trim($ortDylibPath) === '') {
    fail('ORT_DYLIB_PATH must be set before running this smoke test.');
}
if (!is_file($ortDylibPath)) {
    fail("ORT_DYLIB_PATH does not exist: {$ortDylibPath}");
}

// 1) Validation should fail immediately for invalid threshold.
try {
    deepface_compare('a', 'b', 'missing.onnx', 2.0);
    fail('Expected threshold validation error was not thrown.');
} catch (Throwable $e) {
    if (stripos($e->getMessage(), 'threshold') === false) {
        fail("Unexpected threshold validation error: {$e->getMessage()}");
    }
}

// 2) Compare path should fail quickly for a missing model path and never hang.
$missingModel = __DIR__ . '/../missing_model.onnx';
$start = microtime(true);
try {
    deepface_compare('a', 'b', $missingModel, 0.5);
    fail('Expected missing model error was not thrown.');
} catch (Throwable $e) {
    $elapsed = microtime(true) - $start;
    if ($elapsed > 5.0) {
        fail("deepface_compare took too long before failing ({$elapsed}s).");
    }
    echo "SMOKE_OK: compare failed fast in " . round($elapsed, 3) . "s\n";
    echo "SMOKE_OK: error={$e->getMessage()}\n";
}
