# deepfacephp_ext

Rust-based PHP extension prototype for DeepFace-style face comparison using ONNX Runtime, with mandatory face detection + alignment before embedding comparison.

## Prerequisites

- Windows x64 or Linux x64
- Rust nightly toolchain
- PHP CLI (8.3.x recommended)
- ArcFace embedding ONNX model path (passed as `model_path` in `deepface_compare`)
- SCRFD 10G KPS ONNX model path (via `DEEPFACE_DETECTOR_MODEL_PATH`)

## Compare Pipeline

`deepface_compare(img1_path, img2_path, model_path, threshold)` now executes:

1. SCRFD detection
2. strict quality gates (confidence, min face size, blur/sharpness, landmark geometry)
3. 5-point alignment to ArcFace 112x112 template
4. embedding extraction
5. all-pairs similarity scoring with best-pair + margin verification

Verification rule:

- `verified = (best_similarity >= threshold) && (best_similarity - second_best_similarity >= DEEPFACE_PAIR_MARGIN)`

## Runtime Environment Configuration

| Variable | Required | Default | Notes |
|---|---|---:|---|
| `ORT_DYLIB_PATH` | Yes | n/a | Must point to ONNX Runtime shared library (`onnxruntime.dll` on Windows, `libonnxruntime.so` on Linux). |
| `DEEPFACE_DETECTOR_MODEL_PATH` | Yes | n/a | Must point to SCRFD 10G KPS ONNX model file. |
| `DEEPFACE_DETECTOR_INPUT_SIZE` | No | `640` | Must be `>=128` and divisible by `32`. |
| `DEEPFACE_DETECT_CONFIDENCE` | No | `0.80` | Detection confidence gate `[0,1]`. |
| `DEEPFACE_DETECT_NMS_IOU` | No | `0.40` | NMS IoU threshold `[0,1]`. |
| `DEEPFACE_MIN_FACE_SIZE` | No | `96` | Minimum short side of face bbox in pixels. |
| `DEEPFACE_MIN_SHARPNESS` | No | `80.0` | Minimum Laplacian variance threshold. |
| `DEEPFACE_PAIR_MARGIN` | No | `0.06` | Minimum gap between top-1 and top-2 pair scores. |
| `DEEPFACE_DIAGNOSTICS` | No | `0` | `1`/`true`/`yes` to include extra decision diagnostics in response. |

## Local setup and smoke test

Run the helper script from the repository root.

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev-env.ps1
```

Linux:

```bash
chmod +x ./scripts/dev-env.sh
./scripts/dev-env.sh
```

What it does:

1. Downloads ONNX Runtime `1.24.4` into `vendor/onnxruntime` (if missing)
2. Sets `ORT_DYLIB_PATH` to the extracted shared library
3. Runs `cargo check`
4. Runs `cargo build`
5. Runs `scripts/smoke_extension.php`

## Manual smoke command

If `ORT_DYLIB_PATH` is already set and the extension is built.

Windows:

```powershell
php -n -d extension=.\target\debug\deps\deepfacephp_ext.dll .\scripts\smoke_extension.php
```

Linux:

```bash
php -n -d extension=./target/debug/deps/libdeepfacephp_ext.so ./scripts/smoke_extension.php
```

## Error/Fast-Fail Behavior

- Invalid compare `threshold` fails immediately before model loading.
- Missing/invalid `DEEPFACE_DETECTOR_MODEL_PATH` fails immediately before embedder session load.
- If no face passes strict gates, compare fails closed with rejection details in exception text.

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs:

1. `cargo check`
2. `cargo build`
3. PHP smoke test with pinned ONNX Runtime shared library
4. Runs on both `windows-latest` and `ubuntu-latest`
