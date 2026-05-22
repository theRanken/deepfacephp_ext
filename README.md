# deepfacephp_ext

Rust-based PHP extension prototype for DeepFace-style image analysis and face comparison using ONNX Runtime.

## Prerequisites

- Windows x64
- Rust nightly toolchain
- PHP CLI (8.3.x recommended)

## Local setup and smoke test

Run the helper script from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev-env.ps1
```

What it does:

1. Downloads ONNX Runtime `1.24.4` into `vendor/onnxruntime` (if missing)
2. Sets `ORT_DYLIB_PATH` to the extracted `onnxruntime.dll`
3. Runs `cargo check`
4. Runs `cargo build`
5. Runs `scripts/smoke_extension.php`

## Manual smoke command

If `ORT_DYLIB_PATH` is already set and the extension is built:

```powershell
php -n -d extension=.\target\debug\deps\deepfacephp_ext.dll .\scripts\smoke_extension.php
```

## CI

GitHub Actions workflow at `.github/workflows/ci.yml` runs:

1. `cargo check`
2. `cargo build`
3. PHP smoke test with a pinned ONNX Runtime DLL
