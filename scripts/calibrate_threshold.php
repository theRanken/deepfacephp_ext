<?php
declare(strict_types=1);

/**
 * Calibrate threshold from a folder dataset.
 *
 * Supported dataset layouts:
 * 1) Subfolder mode:
 *      root/
 *        person_a/*.jpg
 *        person_b/*.jpg
 *
 * 2) Flat LFW-style mode:
 *      root/
 *        Person_Name_0001.jpg
 *        Person_Name_0002.jpg
 *        Other_Person_0001.jpg
 *
 *    Identity key is inferred as everything before final _NNNN suffix.
 *
 * Usage:
 *   php -n -d extension=... scripts/calibrate_threshold.php \
 *     --dataset "C:/path/to/model_testing" \
 *     --model "C:/path/to/glintr100.onnx" \
 *     --max-positive-per-id 20 \
 *     --negative-ratio 1.0 \
 *     --max-total-pairs 2000 \
 *     --cache-file "./.calibration_cache.json" \
 *     --workers 4 \
 *     --worker-index 0 \
 *     --dump-scores "./scores-w0.json" \
 *     --progress-every 100 \
 *     --step 0.01 \
 *     --seed 42
 */

function fail(string $message, int $code = 1): never
{
    fwrite(STDERR, "CALIBRATE_FAIL: {$message}\n");
    exit($code);
}

function parseArgs(array $argv): array
{
    $opts = getopt('', [
        'dataset:',
        'model:',
        'max-positive-per-id:',
        'negative-ratio:',
        'max-total-pairs:',
        'cache-file:',
        'workers:',
        'worker-index:',
        'dump-scores:',
        'progress-every:',
        'step:',
        'seed:',
    ]);

    if (!isset($opts['dataset']) || trim((string)$opts['dataset']) === '') {
        fail('Missing required --dataset argument.');
    }

    return [
        'dataset' => (string)$opts['dataset'],
        'model' => isset($opts['model']) ? (string)$opts['model'] : '',
        'max_positive_per_id' => isset($opts['max-positive-per-id']) ? (int)$opts['max-positive-per-id'] : 20,
        'negative_ratio' => isset($opts['negative-ratio']) ? (float)$opts['negative-ratio'] : 1.0,
        'max_total_pairs' => isset($opts['max-total-pairs']) ? (int)$opts['max-total-pairs'] : 0,
        'cache_file' => isset($opts['cache-file']) ? (string)$opts['cache-file'] : '',
        'workers' => isset($opts['workers']) ? (int)$opts['workers'] : 1,
        'worker_index' => isset($opts['worker-index']) ? (int)$opts['worker-index'] : 0,
        'dump_scores' => isset($opts['dump-scores']) ? (string)$opts['dump-scores'] : '',
        'progress_every' => isset($opts['progress-every']) ? (int)$opts['progress-every'] : 100,
        'step' => isset($opts['step']) ? (float)$opts['step'] : 0.01,
        'seed' => isset($opts['seed']) ? (int)$opts['seed'] : 42,
    ];
}

function listIdentityImages(string $datasetRoot): array
{
    if (!is_dir($datasetRoot)) {
        fail("Dataset directory does not exist: {$datasetRoot}");
    }

    $fromSubfolders = listIdentityImagesFromSubfolders($datasetRoot);
    if (count($fromSubfolders) >= 2) {
        return $fromSubfolders;
    }

    $fromFlat = listIdentityImagesFromFlatNames($datasetRoot);
    if (count($fromFlat) >= 2) {
        return $fromFlat;
    }

    fail('Could not infer identities. Use subfolders per identity or flat names like Person_Name_0001.jpg.');
}

function listIdentityImagesFromSubfolders(string $datasetRoot): array
{
    $identities = [];
    $dirs = scandir($datasetRoot);
    if ($dirs === false) {
        fail("Unable to read dataset directory: {$datasetRoot}");
    }

    foreach ($dirs as $entry) {
        if ($entry === '.' || $entry === '..') {
            continue;
        }
        $identityDir = rtrim($datasetRoot, "/\\") . DIRECTORY_SEPARATOR . $entry;
        if (!is_dir($identityDir)) {
            continue;
        }
        $images = [];
        $files = scandir($identityDir);
        if ($files === false) {
            continue;
        }
        foreach ($files as $file) {
            if ($file === '.' || $file === '..') {
                continue;
            }
            $path = $identityDir . DIRECTORY_SEPARATOR . $file;
            if (!is_file($path)) {
                continue;
            }
            $ext = strtolower(pathinfo($path, PATHINFO_EXTENSION));
            if (!in_array($ext, ['jpg', 'jpeg', 'png', 'webp', 'bmp'], true)) {
                continue;
            }
            $images[] = str_replace('\\', '/', realpath($path) ?: $path);
        }
        sort($images);
        if (count($images) >= 2) {
            $identities[$entry] = $images;
        }
    }

    return $identities;
}

function listIdentityImagesFromFlatNames(string $datasetRoot): array
{
    $identities = [];
    $files = scandir($datasetRoot);
    if ($files === false) {
        fail("Unable to read dataset directory: {$datasetRoot}");
    }

    foreach ($files as $file) {
        if ($file === '.' || $file === '..') {
            continue;
        }
        $path = rtrim($datasetRoot, "/\\") . DIRECTORY_SEPARATOR . $file;
        if (!is_file($path)) {
            continue;
        }

        if (!preg_match('/^(.+)_([0-9]{4})\.(jpg|jpeg|png|webp|bmp)$/i', $file, $m)) {
            continue;
        }
        $identity = $m[1];
        $identities[$identity] ??= [];
        $identities[$identity][] = str_replace('\\', '/', realpath($path) ?: $path);
    }

    foreach ($identities as $id => $images) {
        sort($images);
        if (count($images) < 2) {
            unset($identities[$id]);
            continue;
        }
        $identities[$id] = $images;
    }

    return $identities;
}

function shardIdentities(array $identities, int $workers, int $workerIndex): array
{
    if ($workers <= 1) {
        return $identities;
    }

    ksort($identities);
    $shard = [];
    $i = 0;
    foreach ($identities as $id => $images) {
        if (($i % $workers) === $workerIndex) {
            $shard[$id] = $images;
        }
        $i++;
    }
    return $shard;
}

function samplePositivePairs(array $images, int $maxPairs, int $seed): array
{
    $pairs = [];
    $n = count($images);
    for ($i = 0; $i < $n; $i++) {
        for ($j = $i + 1; $j < $n; $j++) {
            $pairs[] = [$images[$i], $images[$j]];
        }
    }
    if (count($pairs) <= $maxPairs) {
        return $pairs;
    }
    mt_srand($seed);
    shuffle($pairs);
    return array_slice($pairs, 0, $maxPairs);
}

function sampleNegativePairs(array $identities, int $targetCount, int $seed): array
{
    $pairs = [];
    $idNames = array_keys($identities);
    if (count($idNames) < 2) {
        return $pairs;
    }

    mt_srand($seed);
    $guard = 0;
    $maxGuard = max(1000, $targetCount * 50);
    while (count($pairs) < $targetCount && $guard < $maxGuard) {
        $guard++;
        $idA = $idNames[array_rand($idNames)];
        $idB = $idNames[array_rand($idNames)];
        if ($idA === $idB) {
            continue;
        }
        $imgA = $identities[$idA][array_rand($identities[$idA])];
        $imgB = $identities[$idB][array_rand($identities[$idB])];
        $key = $imgA < $imgB ? "{$imgA}|{$imgB}" : "{$imgB}|{$imgA}";
        if (!isset($pairs[$key])) {
            $pairs[$key] = [$imgA, $imgB];
        }
    }

    return array_values($pairs);
}

function evaluatePair(string $img1, string $img2, string $modelPath): array
{
    $result = deepface_compare($img1, $img2, $modelPath, 0.0);
    if (!is_array($result) || !isset($result['similarity'])) {
        throw new RuntimeException('Invalid compare response shape.');
    }
    return [
        'similarity' => (float)$result['similarity'],
    ];
}

function writeScoresDump(string $path, array $payload): void
{
    $dir = dirname($path);
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    file_put_contents($path, json_encode($payload, JSON_UNESCAPED_SLASHES));
}

function normalizedPathForKey(string $path): string
{
    $resolved = realpath($path);
    if ($resolved !== false) {
        return strtolower(str_replace('\\', '/', $resolved));
    }
    return strtolower(str_replace('\\', '/', $path));
}

function runtimeSignature(string $modelPath): string
{
    $envKeys = [
        'ORT_DYLIB_PATH',
        'DEEPFACE_DETECTOR_MODEL_PATH',
        'DEEPFACE_DETECTOR_INPUT_SIZE',
        'DEEPFACE_DETECT_CONFIDENCE',
        'DEEPFACE_DETECT_NMS_IOU',
        'DEEPFACE_MIN_FACE_SIZE',
        'DEEPFACE_MIN_SHARPNESS',
        'DEEPFACE_PAIR_MARGIN',
    ];
    $parts = ['model=' . normalizedPathForKey($modelPath)];
    foreach ($envKeys as $k) {
        $v = getenv($k);
        $parts[] = $k . '=' . ($v === false ? '' : (string)$v);
    }
    return sha1(implode('|', $parts));
}

function makePairKey(string $img1, string $img2, string $runtimeSig): string
{
    $a = normalizedPathForKey($img1);
    $b = normalizedPathForKey($img2);
    if ($a > $b) {
        [$a, $b] = [$b, $a];
    }
    return sha1($runtimeSig . '|' . $a . '|' . $b);
}

function loadPairCache(string $cachePath): array
{
    if ($cachePath === '' || !is_file($cachePath)) {
        return [];
    }
    $raw = file_get_contents($cachePath);
    if ($raw === false || trim($raw) === '') {
        return [];
    }
    $decoded = json_decode($raw, true);
    if (!is_array($decoded)) {
        return [];
    }
    return $decoded;
}

function savePairCache(string $cachePath, array $cache): void
{
    if ($cachePath === '') {
        return;
    }
    $dir = dirname($cachePath);
    if (!is_dir($dir)) {
        @mkdir($dir, 0777, true);
    }
    file_put_contents($cachePath, json_encode($cache, JSON_UNESCAPED_SLASHES));
}

function evaluatePairsWithCache(
    array $pairs,
    string $modelPath,
    array &$cache,
    string $runtimeSig,
    int $progressEvery,
    string $label,
    array &$stats,
    ?string &$firstError
): array {
    $scores = [];
    $failures = 0;
    $count = count($pairs);
    $i = 0;

    foreach ($pairs as [$img1, $img2]) {
        $i++;
        $key = makePairKey($img1, $img2, $runtimeSig);
        if (isset($cache[$key])) {
            $stats['cache_hits']++;
            $entry = $cache[$key];
            if (($entry['ok'] ?? false) === true && isset($entry['similarity'])) {
                $scores[] = (float)$entry['similarity'];
            } else {
                $failures++;
            }
        } else {
            $stats['cache_misses']++;
            try {
                $similarity = evaluatePair($img1, $img2, $modelPath)['similarity'];
                $scores[] = $similarity;
                $cache[$key] = [
                    'ok' => true,
                    'similarity' => $similarity,
                ];
            } catch (Throwable $e) {
                $failures++;
                if ($firstError === null) {
                    $firstError = $e->getMessage();
                }
                $cache[$key] = [
                    'ok' => false,
                    'error' => $e->getMessage(),
                ];
            }
        }

        if ($progressEvery > 0 && ($i % $progressEvery) === 0) {
            fwrite(STDERR, strtoupper($label) . "_PROGRESS: {$i}/{$count}\n");
            fflush(STDERR);
        }
    }

    return [$scores, $failures];
}

function computeMetrics(array $positives, array $negatives, float $threshold): array
{
    $tp = 0;
    $fn = 0;
    foreach ($positives as $score) {
        if ($score >= $threshold) {
            $tp++;
        } else {
            $fn++;
        }
    }

    $tn = 0;
    $fp = 0;
    foreach ($negatives as $score) {
        if ($score >= $threshold) {
            $fp++;
        } else {
            $tn++;
        }
    }

    $posCount = max(1, count($positives));
    $negCount = max(1, count($negatives));
    $far = $fp / $negCount;
    $frr = $fn / $posCount;
    $acc = ($tp + $tn) / max(1, ($posCount + $negCount));

    return [
        'tp' => $tp,
        'fn' => $fn,
        'tn' => $tn,
        'fp' => $fp,
        'far' => $far,
        'frr' => $frr,
        'acc' => $acc,
    ];
}

function percentile(array $sortedValues, float $p): float
{
    $n = count($sortedValues);
    if ($n === 0) {
        return 0.0;
    }
    if ($n === 1) {
        return (float)$sortedValues[0];
    }
    $idx = ($n - 1) * $p;
    $lo = (int)floor($idx);
    $hi = (int)ceil($idx);
    if ($lo === $hi) {
        return (float)$sortedValues[$lo];
    }
    $w = $idx - $lo;
    return (float)$sortedValues[$lo] * (1.0 - $w) + (float)$sortedValues[$hi] * $w;
}

function fmt(float $v): string
{
    return number_format($v, 4, '.', '');
}

if (!function_exists('deepface_compare')) {
    fail('deepface_compare is not available. Ensure extension is loaded.');
}

$cfg = parseArgs($argv);
$datasetRoot = str_replace('\\', '/', realpath($cfg['dataset']) ?: $cfg['dataset']);
$modelPath = $cfg['model'];

if ($modelPath !== '' && !is_file($modelPath)) {
    fail("Model file does not exist: {$modelPath}");
}

if ($cfg['max_positive_per_id'] < 1) {
    fail('--max-positive-per-id must be >= 1');
}
if ($cfg['negative_ratio'] <= 0) {
    fail('--negative-ratio must be > 0');
}
if ($cfg['step'] <= 0 || $cfg['step'] > 1) {
    fail('--step must be in (0, 1]');
}
if ($cfg['max_total_pairs'] < 0) {
    fail('--max-total-pairs must be >= 0');
}
if ($cfg['progress_every'] < 0) {
    fail('--progress-every must be >= 0');
}
if ($cfg['workers'] < 1) {
    fail('--workers must be >= 1');
}
if ($cfg['worker_index'] < 0 || $cfg['worker_index'] >= $cfg['workers']) {
    fail('--worker-index must be within [0, workers-1]');
}

$identities = listIdentityImages($datasetRoot);
$identities = shardIdentities($identities, $cfg['workers'], $cfg['worker_index']);
if (count($identities) < 2) {
    fail("Worker shard has too few identities for calibration (identities=" . count($identities) . ").");
}

$positivePairs = [];
foreach ($identities as $id => $images) {
    $pairs = samplePositivePairs($images, $cfg['max_positive_per_id'], $cfg['seed'] + crc32((string)$id));
    foreach ($pairs as $pair) {
        $positivePairs[] = $pair;
    }
}

$negativeTarget = (int)max(1, round(count($positivePairs) * $cfg['negative_ratio']));
$negativePairs = sampleNegativePairs($identities, $negativeTarget, $cfg['seed']);

if (count($positivePairs) === 0 || count($negativePairs) === 0) {
    fail('Could not construct enough positive/negative pairs from dataset.');
}

if ($cfg['max_total_pairs'] > 0) {
    $half = max(1, (int)floor($cfg['max_total_pairs'] / 2));
    if (count($positivePairs) > $half) {
        mt_srand($cfg['seed'] + 11);
        shuffle($positivePairs);
        $positivePairs = array_slice($positivePairs, 0, $half);
    }
    if (count($negativePairs) > $half) {
        mt_srand($cfg['seed'] + 17);
        shuffle($negativePairs);
        $negativePairs = array_slice($negativePairs, 0, $half);
    }
}

$cacheFile = $cfg['cache_file'] !== ''
    ? $cfg['cache_file']
    : (rtrim($datasetRoot, "/\\") . '/.calibration_cache.json');
$cache = loadPairCache($cacheFile);
$runtimeSig = runtimeSignature($modelPath);
$cacheStats = ['cache_hits' => 0, 'cache_misses' => 0];
$firstError = null;

fwrite(
    STDERR,
    "CALIBRATION_START: pos_pairs=" . count($positivePairs) .
    " neg_pairs=" . count($negativePairs) .
    " cache_file={$cacheFile}\n"
);
fflush(STDERR);

$positiveScores = [];
$negativeScores = [];
$positiveFailures = 0;
$negativeFailures = 0;
[$positiveScores, $positiveFailures] = evaluatePairsWithCache(
    $positivePairs,
    $modelPath,
    $cache,
    $runtimeSig,
    $cfg['progress_every'],
    'positive',
    $cacheStats,
    $firstError
);
[$negativeScores, $negativeFailures] = evaluatePairsWithCache(
    $negativePairs,
    $modelPath,
    $cache,
    $runtimeSig,
    $cfg['progress_every'],
    'negative',
    $cacheStats,
    $firstError
);

savePairCache($cacheFile, $cache);

if (count($positiveScores) < 10 || count($negativeScores) < 10) {
    $reason = "Too few successful pairs after failures (pos=" . count($positiveScores) . ", neg=" . count($negativeScores) . ").";
    if ($firstError !== null) {
        $reason .= " First error: {$firstError}";
    }
    fail($reason);
}

sort($positiveScores);
sort($negativeScores);

$bestEer = null;
$bestAcc = null;

for ($t = -1.0; $t <= 1.000001; $t += $cfg['step']) {
    $threshold = round($t, 6);
    $metrics = computeMetrics($positiveScores, $negativeScores, $threshold);
    $eerGap = abs($metrics['far'] - $metrics['frr']);

    $row = [
        'threshold' => $threshold,
        'far' => $metrics['far'],
        'frr' => $metrics['frr'],
        'acc' => $metrics['acc'],
        'eer_gap' => $eerGap,
    ];

    if ($bestEer === null || $row['eer_gap'] < $bestEer['eer_gap']) {
        $bestEer = $row;
    }
    if ($bestAcc === null || $row['acc'] > $bestAcc['acc']) {
        $bestAcc = $row;
    }
}

// FAR <= 1% operating point: pick highest threshold that still keeps FAR <= 1%.
$far1 = null;
for ($t = 1.0; $t >= -1.000001; $t -= $cfg['step']) {
    $threshold = round($t, 6);
    $m = computeMetrics($positiveScores, $negativeScores, $threshold);
    if ($m['far'] <= 0.01) {
        $far1 = [
            'threshold' => $threshold,
            'far' => $m['far'],
            'frr' => $m['frr'],
            'acc' => $m['acc'],
        ];
        break;
    }
}

echo "CALIBRATION_DATASET: {$datasetRoot}\n";
echo "MODEL_PATH: " . ($modelPath === '' ? '(bundled resolution)' : $modelPath) . "\n";
echo "WORKERS/INDEX: {$cfg['workers']}/{$cfg['worker_index']}\n";
echo "IDENTITIES_USED: " . count($identities) . "\n";
echo "CACHE_FILE: {$cacheFile}\n";
echo "CACHE_HITS/MISSES: {$cacheStats['cache_hits']}/{$cacheStats['cache_misses']}\n";
echo "PAIRS_USED (pos/neg): " . count($positivePairs) . "/" . count($negativePairs) . "\n";
echo "POSITIVE_PAIRS_SUCCESS: " . count($positiveScores) . " (failures={$positiveFailures})\n";
echo "NEGATIVE_PAIRS_SUCCESS: " . count($negativeScores) . " (failures={$negativeFailures})\n";
echo "POSITIVE_P10/P50/P90: " . fmt(percentile($positiveScores, 0.10)) . " / " . fmt(percentile($positiveScores, 0.50)) . " / " . fmt(percentile($positiveScores, 0.90)) . "\n";
echo "NEGATIVE_P10/P50/P90: " . fmt(percentile($negativeScores, 0.10)) . " / " . fmt(percentile($negativeScores, 0.50)) . " / " . fmt(percentile($negativeScores, 0.90)) . "\n";
echo "\n";
echo "RECOMMENDED (EER-approx): threshold=" . fmt((float)$bestEer['threshold']) .
    " FAR=" . fmt((float)$bestEer['far']) .
    " FRR=" . fmt((float)$bestEer['frr']) .
    " ACC=" . fmt((float)$bestEer['acc']) . "\n";
echo "RECOMMENDED (max-accuracy): threshold=" . fmt((float)$bestAcc['threshold']) .
    " FAR=" . fmt((float)$bestAcc['far']) .
    " FRR=" . fmt((float)$bestAcc['frr']) .
    " ACC=" . fmt((float)$bestAcc['acc']) . "\n";
if ($far1 !== null) {
    echo "RECOMMENDED (FAR<=1%): threshold=" . fmt((float)$far1['threshold']) .
        " FAR=" . fmt((float)$far1['far']) .
        " FRR=" . fmt((float)$far1['frr']) .
        " ACC=" . fmt((float)$far1['acc']) . "\n";
} else {
    echo "RECOMMENDED (FAR<=1%): not found within threshold range\n";
}

if ($cfg['dump_scores'] !== '') {
    writeScoresDump($cfg['dump_scores'], [
        'dataset' => $datasetRoot,
        'model_path' => $modelPath,
        'runtime_signature' => $runtimeSig,
        'workers' => $cfg['workers'],
        'worker_index' => $cfg['worker_index'],
        'positive_scores' => $positiveScores,
        'negative_scores' => $negativeScores,
        'meta' => [
            'identities_used' => count($identities),
            'cache_hits' => $cacheStats['cache_hits'],
            'cache_misses' => $cacheStats['cache_misses'],
            'positive_failures' => $positiveFailures,
            'negative_failures' => $negativeFailures,
        ],
    ]);
    echo "SCORES_DUMP: {$cfg['dump_scores']}\n";
}
