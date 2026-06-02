<?php
declare(strict_types=1);

/**
 * Merge score dumps from parallel calibrate_threshold workers and compute
 * combined threshold recommendations.
 *
 * Usage:
 *   php scripts/merge_calibration_scores.php \
 *     --inputs "scores-w0.json,scores-w1.json,scores-w2.json,scores-w3.json" \
 *     --step 0.01
 */

function fail(string $message, int $code = 1): never
{
    fwrite(STDERR, "MERGE_FAIL: {$message}\n");
    exit($code);
}

function parseArgs(): array
{
    $opts = getopt('', [
        'inputs:',
        'step::',
    ]);

    if (!isset($opts['inputs']) || trim((string)$opts['inputs']) === '') {
        fail('Missing required --inputs argument.');
    }

    $files = array_values(array_filter(array_map('trim', explode(',', (string)$opts['inputs'])), fn($v) => $v !== ''));
    if (count($files) === 0) {
        fail('No input score files provided.');
    }

    $step = isset($opts['step']) ? (float)$opts['step'] : 0.01;
    if ($step <= 0 || $step > 1) {
        fail('--step must be in (0, 1]');
    }

    return ['files' => $files, 'step' => $step];
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
    return [
        'far' => $fp / $negCount,
        'frr' => $fn / $posCount,
        'acc' => ($tp + $tn) / max(1, ($posCount + $negCount)),
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

$cfg = parseArgs();

$positiveScores = [];
$negativeScores = [];
$runtimeSigs = [];
$models = [];
$datasets = [];
$workersRead = 0;

foreach ($cfg['files'] as $file) {
    if (!is_file($file)) {
        fail("Score file does not exist: {$file}");
    }
    $raw = file_get_contents($file);
    if ($raw === false || trim($raw) === '') {
        fail("Score file is empty/unreadable: {$file}");
    }
    $data = json_decode($raw, true);
    if (!is_array($data)) {
        fail("Invalid JSON in score file: {$file}");
    }
    if (!isset($data['positive_scores']) || !isset($data['negative_scores'])) {
        fail("Score file missing required fields: {$file}");
    }

    foreach ((array)$data['positive_scores'] as $s) {
        $positiveScores[] = (float)$s;
    }
    foreach ((array)$data['negative_scores'] as $s) {
        $negativeScores[] = (float)$s;
    }

    if (isset($data['runtime_signature'])) {
        $runtimeSigs[(string)$data['runtime_signature']] = true;
    }
    if (isset($data['model_path'])) {
        $models[(string)$data['model_path']] = true;
    }
    if (isset($data['dataset'])) {
        $datasets[(string)$data['dataset']] = true;
    }
    $workersRead++;
}

if (count($positiveScores) < 10 || count($negativeScores) < 10) {
    fail("Too few aggregated scores (pos=" . count($positiveScores) . ", neg=" . count($negativeScores) . ").");
}

sort($positiveScores);
sort($negativeScores);

$bestEer = null;
$bestAcc = null;
for ($t = -1.0; $t <= 1.000001; $t += $cfg['step']) {
    $threshold = round($t, 6);
    $m = computeMetrics($positiveScores, $negativeScores, $threshold);
    $row = [
        'threshold' => $threshold,
        'far' => $m['far'],
        'frr' => $m['frr'],
        'acc' => $m['acc'],
        'eer_gap' => abs($m['far'] - $m['frr']),
    ];
    if ($bestEer === null || $row['eer_gap'] < $bestEer['eer_gap']) {
        $bestEer = $row;
    }
    if ($bestAcc === null || $row['acc'] > $bestAcc['acc']) {
        $bestAcc = $row;
    }
}

$far1 = null;
for ($t = 1.0; $t >= -1.000001; $t -= $cfg['step']) {
    $threshold = round($t, 6);
    $m = computeMetrics($positiveScores, $negativeScores, $threshold);
    if ($m['far'] <= 0.01) {
        $far1 = ['threshold' => $threshold, 'far' => $m['far'], 'frr' => $m['frr'], 'acc' => $m['acc']];
        break;
    }
}

echo "MERGE_INPUT_FILES: {$workersRead}\n";
echo "DATASETS: " . implode(', ', array_keys($datasets)) . "\n";
echo "MODELS: " . implode(', ', array_keys($models)) . "\n";
echo "RUNTIME_SIGNATURES: " . count($runtimeSigs) . "\n";
if (count($runtimeSigs) > 1) {
    echo "WARN: Multiple runtime signatures detected; merged output may mix different configs.\n";
}
echo "AGG_POSITIVE_SCORES: " . count($positiveScores) . "\n";
echo "AGG_NEGATIVE_SCORES: " . count($negativeScores) . "\n";
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

