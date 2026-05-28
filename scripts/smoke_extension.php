<?php
declare(strict_types=1);

function fail(string $message, int $code = 1): never
{
    fwrite(STDERR, "SMOKE_FAIL: {$message}\n");
    exit($code);
}

if (!function_exists('deepface_analyze') || !function_exists('deepface_compare')) {
    fail('Extension functions are not available. Ensure the extension module is loaded.');
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

// 2) If detector model env is not provided, compare should fail quickly on detector config.
$detectorEnv = getenv('DEEPFACE_DETECTOR_MODEL_PATH');
if ($detectorEnv === false || trim($detectorEnv) === '') {
    $start = microtime(true);
    try {
        deepface_compare('a', 'b', 'missing.onnx', 0.5);
        fail('Expected missing detector config error was not thrown.');
    } catch (Throwable $e) {
        $elapsed = microtime(true) - $start;
        if ($elapsed > 5.0) {
            fail("deepface_compare took too long before detector-config failure ({$elapsed}s).");
        }
        if (stripos($e->getMessage(), 'DEEPFACE_DETECTOR_MODEL_PATH') === false && stripos($e->getMessage(), 'detector') === false) {
            fail("Unexpected detector-config error: {$e->getMessage()}");
        }
        echo "SMOKE_OK: compare failed fast for detector config in " . round($elapsed, 3) . "s\n";
        echo "SMOKE_OK: error={$e->getMessage()}\n";
    }
} else {
    echo "SMOKE_SKIP: DEEPFACE_DETECTOR_MODEL_PATH is already set; skipping missing-detector check\n";
}

// 3) If detector model env exists and file is valid, compare should still fail quickly for missing embedder model.
if ($detectorEnv !== false && trim($detectorEnv) !== '' && is_file($detectorEnv)) {
    $missingEmbedder = __DIR__ . '/../missing_model.onnx';
    $start = microtime(true);
    try {
        deepface_compare('a', 'b', $missingEmbedder, 0.5);
        fail('Expected missing embedder model error was not thrown.');
    } catch (Throwable $e) {
        $elapsed = microtime(true) - $start;
        if ($elapsed > 5.0) {
            fail("deepface_compare took too long before embedder-model failure ({$elapsed}s).");
        }
        if (stripos($e->getMessage(), 'missing_model.onnx') === false && stripos($e->getMessage(), 'does not exist') === false) {
            fail("Unexpected embedder-model error: {$e->getMessage()}");
        }
        echo "SMOKE_OK: compare failed fast for missing embedder model in " . round($elapsed, 3) . "s\n";
        echo "SMOKE_OK: error={$e->getMessage()}\n";
    }
}
