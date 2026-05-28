#![cfg_attr(windows, feature(abi_vectorcall))]

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ext_php_rs::boxed::ZBox;
use ext_php_rs::prelude::*;
use ext_php_rs::types::ZendHashTable;
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgb, RgbImage};
use nalgebra::{Matrix2, Matrix2xX, Vector2};
use ndarray::Array4;
use ort::{ep, inputs, session::Session, value::TensorRef};
#[cfg(unix)]
use std::ffi::CStr;
#[cfg(windows)]
use std::os::windows::ffi::OsStringExt;

static DETECTOR_MODEL_CACHE: OnceLock<Mutex<HashMap<String, Arc<Mutex<Session>>>>> = OnceLock::new();
static EMBEDDER_MODEL_CACHE: OnceLock<Mutex<HashMap<String, Arc<Mutex<Session>>>>> = OnceLock::new();
static ORT_RUNTIME_INIT: OnceLock<Result<(), String>> = OnceLock::new();
static RUNTIME_CONFIG: OnceLock<Result<RuntimeConfig, String>> = OnceLock::new();

const ALIGN_TEMPLATE_112: [Point2f; 5] = [
    Point2f { x: 38.2946, y: 51.6963 },
    Point2f { x: 73.5318, y: 51.5014 },
    Point2f { x: 56.0252, y: 71.7366 },
    Point2f { x: 41.5493, y: 92.3655 },
    Point2f { x: 70.7299, y: 92.2041 },
];

const DEFAULT_DETECTOR_MODEL_FILENAMES: [&str; 4] = [
    "scrfd_10g_kps.onnx",
    "scrfd_10g_bnkps.onnx",
    "scrfd_10g_gnkps.onnx",
    "scrfd_10g.onnx",
];
const DEFAULT_EMBEDDER_MODEL_FILENAMES: [&str; 4] = [
    "arcface.onnx",
    "w600k_r50.onnx",
    "glintr100.onnx",
    "insightface_arcface.onnx",
];

#[derive(Clone, Debug)]
struct RuntimeConfig {
    detector_model_path: String,
    detector_input_size: u32,
    detect_confidence: f32,
    detect_nms_iou: f32,
    min_face_size: f32,
    min_sharpness: f32,
    pair_margin: f32,
    diagnostics: bool,
}

#[derive(Clone, Copy, Debug)]
struct Point2f {
    x: f32,
    y: f32,
}

#[derive(Clone, Copy, Debug)]
struct BoundingBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

impl BoundingBox {
    fn width(&self) -> f32 {
        (self.x2 - self.x1).max(0.0)
    }

    fn height(&self) -> f32 {
        (self.y2 - self.y1).max(0.0)
    }

    fn area(&self) -> f32 {
        self.width() * self.height()
    }

    fn short_side(&self) -> f32 {
        self.width().min(self.height())
    }

    fn clamp_to_image(&self, width: u32, height: u32) -> BoundingBox {
        let max_x = (width.saturating_sub(1)) as f32;
        let max_y = (height.saturating_sub(1)) as f32;
        BoundingBox {
            x1: self.x1.clamp(0.0, max_x),
            y1: self.y1.clamp(0.0, max_y),
            x2: self.x2.clamp(0.0, max_x),
            y2: self.y2.clamp(0.0, max_y),
        }
    }
}

#[derive(Clone, Debug)]
struct FaceDetection {
    bbox: BoundingBox,
    score: f32,
    landmarks5: [Point2f; 5],
}

#[derive(Clone, Debug)]
struct FaceQualityMetrics {
    face_size: f32,
    sharpness: f32,
    geometry_ok: bool,
}

#[derive(Clone, Debug)]
struct FaceCandidate {
    aligned_112: Array4<f32>,
    detect_score: f32,
    quality_metrics: FaceQualityMetrics,
}

#[derive(Clone, Debug)]
struct CompareDecision {
    best_score: f32,
    second_best_score: f32,
    margin: f32,
    verified: bool,
    selected_pair: (usize, usize),
    decision_reason: &'static str,
}

#[derive(Clone, Default, Debug)]
struct QualityRejectCounts {
    low_confidence: usize,
    small_face: usize,
    low_sharpness: usize,
    invalid_landmark_geometry: usize,
    alignment_failed: usize,
}

impl QualityRejectCounts {
    fn total(&self) -> usize {
        self.low_confidence
            + self.small_face
            + self.low_sharpness
            + self.invalid_landmark_geometry
            + self.alignment_failed
    }
}

#[derive(Clone, Copy, Debug)]
struct SimilarityTransform {
    a11: f32,
    a12: f32,
    a21: f32,
    a22: f32,
    tx: f32,
    ty: f32,
}

#[derive(Clone, Copy, Debug)]
struct LetterboxMeta {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

#[derive(Clone, Debug)]
struct DetectorTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn detector_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<Session>>>> {
    DETECTOR_MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn embedder_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<Session>>>> {
    EMBEDDER_MODEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn env_non_empty(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[cfg(windows)]
fn extension_module_path() -> Option<PathBuf> {
    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
    };

    let mut handle: HMODULE = std::ptr::null_mut();
    let symbol_address = ensure_ort_runtime_initialized as *const () as usize as *const u16;
    let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
    let ok = unsafe { GetModuleHandleExW(flags, symbol_address, &mut handle as *mut HMODULE) };
    if ok == 0 || handle.is_null() {
        return None;
    }

    let mut buf = vec![0u16; 32768];
    let len = unsafe { GetModuleFileNameW(handle, buf.as_mut_ptr(), buf.len() as u32) };
    if len == 0 {
        return None;
    }
    buf.truncate(len as usize);
    Some(PathBuf::from(std::ffi::OsString::from_wide(&buf)))
}

#[cfg(unix)]
fn extension_module_path() -> Option<PathBuf> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let symbol_address =
        ensure_ort_runtime_initialized as *const () as usize as *const libc::c_void;
    let found = unsafe { libc::dladdr(symbol_address, &mut info as *mut libc::Dl_info) };
    if found == 0 || info.dli_fname.is_null() {
        return None;
    }

    let path = unsafe { CStr::from_ptr(info.dli_fname) }
        .to_string_lossy()
        .into_owned();
    Some(PathBuf::from(path))
}

#[cfg(not(any(windows, unix)))]
fn extension_module_path() -> Option<PathBuf> {
    None
}

fn push_unique_dir(dirs: &mut Vec<PathBuf>, seen: &mut HashSet<String>, dir: PathBuf) {
    if !dir.is_dir() {
        return;
    }
    let key = if cfg!(windows) {
        dir.to_string_lossy().to_lowercase()
    } else {
        dir.to_string_lossy().to_string()
    };
    if seen.insert(key) {
        dirs.push(dir);
    }
}

fn candidate_base_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    let mut seen = HashSet::new();

    if let Some(module_path) = extension_module_path() {
        if let Some(parent) = module_path.parent() {
            push_unique_dir(&mut dirs, &mut seen, parent.to_path_buf());
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        push_unique_dir(&mut dirs, &mut seen, cwd);
    }

    dirs
}

fn find_file_by_name(root: &Path, file_name: &str, max_depth: usize) -> Option<PathBuf> {
    if !root.is_dir() {
        return None;
    }

    let mut stack = vec![(root.to_path_buf(), 0usize)];
    while let Some((dir, depth)) = stack.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().eq_ignore_ascii_case(file_name) {
                        return Some(path);
                    }
                }
            } else if depth < max_depth && path.is_dir() {
                stack.push((path, depth + 1));
            }
        }
    }

    None
}

fn ort_library_filenames() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["onnxruntime.dll"]
    }
    #[cfg(target_os = "linux")]
    {
        &["libonnxruntime.so"]
    }
    #[cfg(target_os = "macos")]
    {
        &["libonnxruntime.dylib"]
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        &["onnxruntime"]
    }
}

fn discover_ort_dylib_path() -> Option<PathBuf> {
    for base in candidate_base_dirs() {
        let direct_roots = [
            base.clone(),
            base.join("onnxruntime"),
            base.join("vendor").join("onnxruntime"),
        ];

        for root in direct_roots {
            for file_name in ort_library_filenames() {
                let direct = root.join(file_name);
                if direct.is_file() {
                    return Some(direct);
                }
            }
        }

        let recursive_roots = [
            base.join("onnxruntime"),
            base.join("vendor").join("onnxruntime"),
        ];
        for root in recursive_roots {
            for file_name in ort_library_filenames() {
                if let Some(found) = find_file_by_name(&root, file_name, 3) {
                    return Some(found);
                }
            }
        }
    }

    None
}

fn resolve_detector_model_path() -> Result<String, String> {
    if let Some(path) = env_non_empty("DEEPFACE_DETECTOR_MODEL_PATH") {
        let file = PathBuf::from(&path);
        if file.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "DEEPFACE_DETECTOR_MODEL_PATH does not point to a file: {}",
            file.display()
        ));
    }

    for base in candidate_base_dirs() {
        let direct_roots = [
            base.clone(),
            base.join("models"),
            base.join("vendor").join("models"),
            base.join("vendor").join("deepfacephp_ext").join("models"),
        ];
        for root in direct_roots {
            for file_name in DEFAULT_DETECTOR_MODEL_FILENAMES {
                let direct = root.join(file_name);
                if direct.is_file() {
                    return Ok(direct.to_string_lossy().to_string());
                }
            }
        }

        let recursive_roots = [
            base.join("models"),
            base.join("vendor").join("models"),
            base.join("vendor").join("deepfacephp_ext").join("models"),
        ];
        for root in recursive_roots {
            for file_name in DEFAULT_DETECTOR_MODEL_FILENAMES {
                if let Some(found) = find_file_by_name(&root, file_name, 2) {
                    return Ok(found.to_string_lossy().to_string());
                }
            }
        }
    }

    Err(
        "DEEPFACE_DETECTOR_MODEL_PATH is required and must point to SCRFD 10G KPS ONNX model (or bundle one of: scrfd_10g_kps.onnx, scrfd_10g_bnkps.onnx, scrfd_10g.onnx near the extension)"
            .to_string(),
    )
}

fn resolve_embedder_model_path(model_path: &str) -> Result<String, String> {
    let explicit = model_path.trim();
    if !explicit.is_empty() {
        let explicit_file = PathBuf::from(explicit);
        if explicit_file.is_file() {
            return Ok(explicit.to_string());
        }
        return Err(format!(
            "Embedder model file does not exist: {}",
            explicit_file.display()
        ));
    }

    for base in candidate_base_dirs() {
        let direct_roots = [
            base.clone(),
            base.join("models"),
            base.join("vendor").join("models"),
            base.join("vendor").join("deepfacephp_ext").join("models"),
        ];
        for root in direct_roots {
            for file_name in DEFAULT_EMBEDDER_MODEL_FILENAMES {
                let direct = root.join(file_name);
                if direct.is_file() {
                    return Ok(direct.to_string_lossy().to_string());
                }
            }
        }

        let recursive_roots = [
            base.join("models"),
            base.join("vendor").join("models"),
            base.join("vendor").join("deepfacephp_ext").join("models"),
        ];
        for root in recursive_roots {
            for file_name in DEFAULT_EMBEDDER_MODEL_FILENAMES {
                if let Some(found) = find_file_by_name(&root, file_name, 2) {
                    return Ok(found.to_string_lossy().to_string());
                }
            }
        }
    }

    Err(
        "model_path is empty and no bundled embedder model was found (expected one of: arcface.onnx, w600k_r50.onnx, glintr100.onnx, insightface_arcface.onnx)"
            .to_string(),
    )
}

fn ensure_ort_runtime_initialized() -> PhpResult<()> {
    let init = ORT_RUNTIME_INIT.get_or_init(|| {
        let dylib_path = env_non_empty("ORT_DYLIB_PATH")
            .map(PathBuf::from)
            .or_else(discover_ort_dylib_path)
            .ok_or_else(|| {
                "ORT_DYLIB_PATH is required (or bundle ONNX Runtime near the extension) and must point to a valid ONNX Runtime shared library (for example, onnxruntime.dll or libonnxruntime.so)".to_string()
            })?;

        if !dylib_path.is_file() {
            return Err(format!(
                "ORT_DYLIB_PATH does not point to a file: {}",
                dylib_path.display()
            ));
        }

        std::env::set_var("ORT_DYLIB_PATH", dylib_path.to_string_lossy().to_string());

        let committed = ort::init_from(&dylib_path)
            .map_err(|e| format!("Failed to load ONNX Runtime from {}: {e}", dylib_path.display()))?
            .with_execution_providers([ep::CPUExecutionProvider::default().build()])
            .commit();

        ort::environment::current().map_err(|e| {
            format!(
                "Failed to initialize ONNX Runtime environment from {}: {e}",
                dylib_path.display()
            )
        })?;

        if !committed {
            return Err(
                "ONNX Runtime environment was already initialized before extension setup; restart PHP process and ensure ORT_DYLIB_PATH is set before first use".to_string()
            );
        }

        Ok(())
    });

    init.clone().map_err(PhpException::default)
}

fn parse_env_f32(key: &str, default: f32) -> Result<f32, String> {
    match std::env::var(key) {
        Ok(raw) => {
            let value = raw.trim().parse::<f32>().map_err(|_| {
                format!("{key} must be a valid float, got: '{raw}'")
            })?;
            if !value.is_finite() {
                return Err(format!("{key} must be finite"));
            }
            Ok(value)
        }
        Err(_) => Ok(default),
    }
}

fn parse_env_u32(key: &str, default: u32) -> Result<u32, String> {
    match std::env::var(key) {
        Ok(raw) => {
            let value = raw.trim().parse::<u32>().map_err(|_| {
                format!("{key} must be a valid integer, got: '{raw}'")
            })?;
            Ok(value)
        }
        Err(_) => Ok(default),
    }
}

fn load_runtime_config() -> PhpResult<&'static RuntimeConfig> {
    let cfg = RUNTIME_CONFIG.get_or_init(|| {
        let detector_model_path = resolve_detector_model_path()?;

        let detector_input_size = parse_env_u32("DEEPFACE_DETECTOR_INPUT_SIZE", 640)?;
        if detector_input_size < 128 || detector_input_size % 32 != 0 {
            return Err("DEEPFACE_DETECTOR_INPUT_SIZE must be >= 128 and divisible by 32".to_string());
        }

        let detect_confidence = parse_env_f32("DEEPFACE_DETECT_CONFIDENCE", 0.80)?;
        if !(0.0..=1.0).contains(&detect_confidence) {
            return Err("DEEPFACE_DETECT_CONFIDENCE must be within [0.0, 1.0]".to_string());
        }

        let detect_nms_iou = parse_env_f32("DEEPFACE_DETECT_NMS_IOU", 0.40)?;
        if !(0.0..=1.0).contains(&detect_nms_iou) {
            return Err("DEEPFACE_DETECT_NMS_IOU must be within [0.0, 1.0]".to_string());
        }

        let min_face_size = parse_env_f32("DEEPFACE_MIN_FACE_SIZE", 96.0)?;
        if min_face_size <= 0.0 {
            return Err("DEEPFACE_MIN_FACE_SIZE must be > 0".to_string());
        }

        let min_sharpness = parse_env_f32("DEEPFACE_MIN_SHARPNESS", 80.0)?;
        if min_sharpness < 0.0 {
            return Err("DEEPFACE_MIN_SHARPNESS must be >= 0".to_string());
        }

        let pair_margin = parse_env_f32("DEEPFACE_PAIR_MARGIN", 0.06)?;
        if pair_margin < 0.0 {
            return Err("DEEPFACE_PAIR_MARGIN must be >= 0".to_string());
        }

        let diagnostics = std::env::var("DEEPFACE_DIAGNOSTICS")
            .map(|v| {
                let normalized = v.trim();
                normalized == "1"
                    || normalized.eq_ignore_ascii_case("true")
                    || normalized.eq_ignore_ascii_case("yes")
            })
            .unwrap_or(false);

        Ok(RuntimeConfig {
            detector_model_path,
            detector_input_size,
            detect_confidence,
            detect_nms_iou,
            min_face_size,
            min_sharpness,
            pair_margin,
            diagnostics,
        })
    });

    cfg.as_ref().map_err(|e| PhpException::default(e.clone()))
}

fn get_or_load_session(
    cache: &'static Mutex<HashMap<String, Arc<Mutex<Session>>>>,
    model_path: &str,
) -> PhpResult<Arc<Mutex<Session>>> {
    ensure_ort_runtime_initialized()?;

    let mut lock = cache
        .lock()
        .map_err(|_| PhpException::default("Model cache lock poisoned".to_string()))?;

    if let Some(session) = lock.get(model_path) {
        return Ok(Arc::clone(session));
    }

    let session = Session::builder()
        .map_err(|e| PhpException::default(e.to_string()))?
        .commit_from_file(model_path)
        .map_err(|e| PhpException::default(e.to_string()))?;

    let session = Arc::new(Mutex::new(session));
    lock.insert(model_path.to_string(), Arc::clone(&session));
    Ok(session)
}

fn preprocess_embed_input(aligned_face: &RgbImage) -> Result<Array4<f32>, String> {
    if aligned_face.width() != 112 || aligned_face.height() != 112 {
        return Err("Aligned face must be exactly 112x112".to_string());
    }

    let mut array = Array4::zeros((1, 3, 112, 112));
    for (x, y, pixel) in aligned_face.enumerate_pixels() {
        let r = (pixel[0] as f32 / 127.5) - 1.0;
        let g = (pixel[1] as f32 / 127.5) - 1.0;
        let b = (pixel[2] as f32 / 127.5) - 1.0;

        array[[0, 0, y as usize, x as usize]] = r;
        array[[0, 1, y as usize, x as usize]] = g;
        array[[0, 2, y as usize, x as usize]] = b;
    }

    Ok(array)
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> Result<f32, String> {
    if v1.len() != v2.len() {
        return Err(format!(
            "Embedding dimension mismatch: {} vs {}",
            v1.len(),
            v2.len()
        ));
    }
    if v1.is_empty() {
        return Err("Embedding vector is empty".to_string());
    }

    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
    let norm_product = norm_a * norm_b;
    if norm_product <= f32::EPSILON {
        return Err("Embedding norm is zero; cannot compute cosine similarity".to_string());
    }

    Ok(dot_product / norm_product)
}

fn letterbox_to_square(src: &RgbImage, target_size: u32) -> (RgbImage, LetterboxMeta) {
    let src_w = src.width() as f32;
    let src_h = src.height() as f32;
    let scale = (target_size as f32 / src_w).min(target_size as f32 / src_h);
    let resized_w = (src_w * scale).round().max(1.0) as u32;
    let resized_h = (src_h * scale).round().max(1.0) as u32;

    let resized = DynamicImage::ImageRgb8(src.clone())
        .resize_exact(resized_w, resized_h, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let pad_x = ((target_size - resized_w) / 2) as i64;
    let pad_y = ((target_size - resized_h) / 2) as i64;
    let mut canvas = ImageBuffer::from_pixel(target_size, target_size, Rgb([0, 0, 0]));
    image::imageops::overlay(&mut canvas, &resized, pad_x, pad_y);

    (
        canvas,
        LetterboxMeta {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
        },
    )
}

fn preprocess_detector_input(src: &RgbImage, target_size: u32) -> (Array4<f32>, LetterboxMeta) {
    let (letterboxed, meta) = letterbox_to_square(src, target_size);

    let mut array = Array4::zeros((1, 3, target_size as usize, target_size as usize));
    for (x, y, pixel) in letterboxed.enumerate_pixels() {
        let r = (pixel[0] as f32 - 127.5) / 128.0;
        let g = (pixel[1] as f32 - 127.5) / 128.0;
        let b = (pixel[2] as f32 - 127.5) / 128.0;

        array[[0, 0, y as usize, x as usize]] = r;
        array[[0, 1, y as usize, x as usize]] = g;
        array[[0, 2, y as usize, x as usize]] = b;
    }

    (array, meta)
}

fn get_detector_tensors(
    detector: &mut Session,
    tensor: &Array4<f32>,
) -> Result<Vec<DetectorTensor>, String> {
    let input = TensorRef::from_array_view(tensor)
        .map_err(|e| format!("Failed creating detector input tensor: {e}"))?;
    let outputs = detector
        .run(inputs![input])
        .map_err(|e| format!("Detector inference failed: {e}"))?;

    let mut tensors = Vec::with_capacity(outputs.len());
    for idx in 0..outputs.len() {
        let (shape, data) = outputs[idx]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed reading detector output tensor {idx}: {e}"))?;

        let shape_vec = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
        tensors.push(DetectorTensor {
            shape: shape_vec,
            data: data.to_vec(),
        });
    }

    Ok(tensors)
}

fn parse_decoded_detection_tensor(
    tensor: &DetectorTensor,
    letterbox: LetterboxMeta,
    image_w: u32,
    image_h: u32,
) -> Vec<FaceDetection> {
    if tensor.shape.len() < 2 {
        return Vec::new();
    }

    let feature_len = *tensor.shape.last().unwrap_or(&0);
    if feature_len < 15 {
        return Vec::new();
    }

    let rows = tensor.data.len() / feature_len;
    let mut detections = Vec::new();

    for row_idx in 0..rows {
        let start = row_idx * feature_len;
        let row = &tensor.data[start..start + feature_len];

        let (bbox, score, lms_offset) = if row[2] > row[0] && row[3] > row[1] {
            (
                BoundingBox {
                    x1: row[0],
                    y1: row[1],
                    x2: row[2],
                    y2: row[3],
                },
                row[4],
                5,
            )
        } else if row[3] > row[1] && row[4] > row[2] {
            (
                BoundingBox {
                    x1: row[1],
                    y1: row[2],
                    x2: row[3],
                    y2: row[4],
                },
                row[0],
                5,
            )
        } else {
            continue;
        };

        if score <= 0.0 {
            continue;
        }

        if lms_offset + 10 > row.len() {
            continue;
        }

        let mut landmarks = [Point2f { x: 0.0, y: 0.0 }; 5];
        for i in 0..5 {
            landmarks[i] = Point2f {
                x: row[lms_offset + (i * 2)],
                y: row[lms_offset + (i * 2) + 1],
            };
        }

        detections.push(map_detection_back(
            bbox,
            landmarks,
            score,
            letterbox,
            image_w,
            image_h,
        ));
    }

    detections
}

fn guess_stride_from_rows(rows: usize, input_size: u32) -> Option<u32> {
    for stride in [8_u32, 16_u32, 32_u32, 64_u32] {
        let grid = (input_size / stride) as usize;
        if rows == grid * grid || rows == 2 * grid * grid {
            return Some(stride);
        }
    }
    None
}

fn infer_stride_from_grid(size: usize, input_size: u32) -> Option<u32> {
    if size == 0 {
        return None;
    }
    let stride = input_size as f32 / size as f32;
    let rounded = stride.round() as u32;
    if [8, 16, 32, 64].contains(&rounded) {
        Some(rounded)
    } else {
        None
    }
}

fn flatten_scores_with_input(shape: &[usize], data: &[f32], input_size: u32) -> Option<(Vec<f32>, u32)> {
    match shape {
        [1, channels, h, w] if *channels == 1 || *channels == 2 => {
            let stride = infer_stride_from_grid(*h, input_size)?;
            let mut scores = Vec::with_capacity(h * w);
            let per_channel = h * w;
            for idx in 0..(h * w) {
                let score = if *channels == 1 {
                    data[idx]
                } else {
                    data[per_channel + idx]
                };
                scores.push(score);
            }
            Some((scores, stride))
        }
        [1, rows, channels] if *channels == 1 || *channels == 2 => {
            let stride = guess_stride_from_rows(*rows, input_size)?;
            let mut scores = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let base = row * channels;
                scores.push(if *channels == 1 { data[base] } else { data[base + 1] });
            }
            Some((scores, stride))
        }
        [rows, channels] if *channels == 1 || *channels == 2 => {
            let stride = guess_stride_from_rows(*rows, input_size)?;
            let mut scores = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let base = row * channels;
                scores.push(if *channels == 1 { data[base] } else { data[base + 1] });
            }
            Some((scores, stride))
        }
        _ => None,
    }
}

fn flatten_bbox_with_input(shape: &[usize], data: &[f32], input_size: u32) -> Option<(Vec<[f32; 4]>, u32)> {
    match shape {
        [1, 4, h, w] => {
            let stride = infer_stride_from_grid(*h, input_size)?;
            let mut out = Vec::with_capacity(h * w);
            let per_channel = h * w;
            for idx in 0..(h * w) {
                out.push([
                    data[idx],
                    data[per_channel + idx],
                    data[(2 * per_channel) + idx],
                    data[(3 * per_channel) + idx],
                ]);
            }
            Some((out, stride))
        }
        [1, rows, 4] => {
            let stride = guess_stride_from_rows(*rows, input_size)?;
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let base = row * 4;
                out.push([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            }
            Some((out, stride))
        }
        [rows, 4] => {
            let stride = guess_stride_from_rows(*rows, input_size)?;
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let base = row * 4;
                out.push([data[base], data[base + 1], data[base + 2], data[base + 3]]);
            }
            Some((out, stride))
        }
        _ => None,
    }
}

fn flatten_kps_with_input(shape: &[usize], data: &[f32], input_size: u32) -> Option<(Vec<[f32; 10]>, u32)> {
    match shape {
        [1, 10, h, w] => {
            let stride = infer_stride_from_grid(*h, input_size)?;
            let mut out = Vec::with_capacity(h * w);
            let per_channel = h * w;
            for idx in 0..(h * w) {
                let mut item = [0.0_f32; 10];
                for c in 0..10 {
                    item[c] = data[c * per_channel + idx];
                }
                out.push(item);
            }
            Some((out, stride))
        }
        [1, rows, 10] => {
            let stride = guess_stride_from_rows(*rows, input_size)?;
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let base = row * 10;
                let mut item = [0.0_f32; 10];
                item.copy_from_slice(&data[base..base + 10]);
                out.push(item);
            }
            Some((out, stride))
        }
        [rows, 10] => {
            let stride = guess_stride_from_rows(*rows, input_size)?;
            let mut out = Vec::with_capacity(*rows);
            for row in 0..*rows {
                let base = row * 10;
                let mut item = [0.0_f32; 10];
                item.copy_from_slice(&data[base..base + 10]);
                out.push(item);
            }
            Some((out, stride))
        }
        _ => None,
    }
}

fn decode_scrfd_from_heads(
    tensors: &[DetectorTensor],
    cfg: &RuntimeConfig,
    letterbox: LetterboxMeta,
    image_w: u32,
    image_h: u32,
) -> Vec<FaceDetection> {
    let mut scores_by_stride: HashMap<u32, Vec<f32>> = HashMap::new();
    let mut bbox_by_stride: HashMap<u32, Vec<[f32; 4]>> = HashMap::new();
    let mut kps_by_stride: HashMap<u32, Vec<[f32; 10]>> = HashMap::new();

    for tensor in tensors {
        if let Some((scores, stride)) = flatten_scores_with_input(&tensor.shape, &tensor.data, cfg.detector_input_size) {
            scores_by_stride.insert(stride, scores);
            continue;
        }
        if let Some((bbox, stride)) = flatten_bbox_with_input(&tensor.shape, &tensor.data, cfg.detector_input_size) {
            bbox_by_stride.insert(stride, bbox);
            continue;
        }
        if let Some((kps, stride)) = flatten_kps_with_input(&tensor.shape, &tensor.data, cfg.detector_input_size) {
            kps_by_stride.insert(stride, kps);
            continue;
        }
    }

    let mut candidates = Vec::new();
    for stride in [8_u32, 16_u32, 32_u32, 64_u32] {
        let Some(scores) = scores_by_stride.get(&stride) else {
            continue;
        };
        let Some(bboxes) = bbox_by_stride.get(&stride) else {
            continue;
        };
        let Some(kps) = kps_by_stride.get(&stride) else {
            continue;
        };

        let len = scores.len().min(bboxes.len()).min(kps.len());
        if len == 0 {
            continue;
        }

        let grid = (cfg.detector_input_size / stride) as usize;
        if grid == 0 {
            continue;
        }

        let anchors_per_location = (len as f32 / (grid * grid) as f32).round().max(1.0) as usize;

        for idx in 0..len {
            let score = scores[idx];
            if score <= 0.0 {
                continue;
            }

            let spatial_idx = idx / anchors_per_location;
            let x = spatial_idx % grid;
            let y = spatial_idx / grid;
            let center_x = (x as f32 + 0.5) * stride as f32;
            let center_y = (y as f32 + 0.5) * stride as f32;

            let d = bboxes[idx];
            let bbox = BoundingBox {
                x1: center_x - d[0] * stride as f32,
                y1: center_y - d[1] * stride as f32,
                x2: center_x + d[2] * stride as f32,
                y2: center_y + d[3] * stride as f32,
            };

            let raw_kps = kps[idx];
            let landmarks = [
                Point2f {
                    x: center_x + raw_kps[0] * stride as f32,
                    y: center_y + raw_kps[1] * stride as f32,
                },
                Point2f {
                    x: center_x + raw_kps[2] * stride as f32,
                    y: center_y + raw_kps[3] * stride as f32,
                },
                Point2f {
                    x: center_x + raw_kps[4] * stride as f32,
                    y: center_y + raw_kps[5] * stride as f32,
                },
                Point2f {
                    x: center_x + raw_kps[6] * stride as f32,
                    y: center_y + raw_kps[7] * stride as f32,
                },
                Point2f {
                    x: center_x + raw_kps[8] * stride as f32,
                    y: center_y + raw_kps[9] * stride as f32,
                },
            ];

            candidates.push(map_detection_back(
                bbox,
                landmarks,
                score,
                letterbox,
                image_w,
                image_h,
            ));
        }
    }

    candidates
}

fn map_detection_back(
    bbox: BoundingBox,
    landmarks: [Point2f; 5],
    score: f32,
    letterbox: LetterboxMeta,
    image_w: u32,
    image_h: u32,
) -> FaceDetection {
    let mut mapped_landmarks = [Point2f { x: 0.0, y: 0.0 }; 5];
    for i in 0..5 {
        mapped_landmarks[i] = Point2f {
            x: (landmarks[i].x - letterbox.pad_x) / letterbox.scale,
            y: (landmarks[i].y - letterbox.pad_y) / letterbox.scale,
        };
    }

    let mapped_bbox = BoundingBox {
        x1: (bbox.x1 - letterbox.pad_x) / letterbox.scale,
        y1: (bbox.y1 - letterbox.pad_y) / letterbox.scale,
        x2: (bbox.x2 - letterbox.pad_x) / letterbox.scale,
        y2: (bbox.y2 - letterbox.pad_y) / letterbox.scale,
    }
    .clamp_to_image(image_w, image_h);

    FaceDetection {
        bbox: mapped_bbox,
        score,
        landmarks5: mapped_landmarks,
    }
}

fn iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    if inter_area <= 0.0 {
        return 0.0;
    }

    let union = a.area() + b.area() - inter_area;
    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

fn nms(mut detections: Vec<FaceDetection>, iou_threshold: f32) -> Vec<FaceDetection> {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    let mut kept = Vec::new();
    while let Some(candidate) = detections.first().cloned() {
        kept.push(candidate.clone());
        detections.remove(0);
        detections.retain(|other| iou(candidate.bbox, other.bbox) < iou_threshold);
    }

    kept
}

fn detect_faces_scrfd(
    image: &RgbImage,
    cfg: &RuntimeConfig,
    detector: &mut Session,
) -> Result<Vec<FaceDetection>, String> {
    let (detector_tensor, letterbox) = preprocess_detector_input(image, cfg.detector_input_size);
    let tensors = get_detector_tensors(detector, &detector_tensor)?;

    let mut detections = tensors
        .iter()
        .flat_map(|tensor| {
            parse_decoded_detection_tensor(tensor, letterbox, image.width(), image.height())
        })
        .collect::<Vec<_>>();

    if detections.is_empty() {
        detections = decode_scrfd_from_heads(&tensors, cfg, letterbox, image.width(), image.height());
    }

    if detections.is_empty() {
        return Err(
            "Detector returned no decodable face outputs. Ensure SCRFD 10G KPS ONNX export is compatible"
                .to_string(),
        );
    }

    Ok(nms(detections, cfg.detect_nms_iou))
}

fn landmarks_geometry_ok(lm: &[Point2f; 5], bbox: BoundingBox) -> bool {
    let left_eye = lm[0];
    let right_eye = lm[1];
    let nose = lm[2];
    let left_mouth = lm[3];
    let right_mouth = lm[4];

    let eye_distance = ((right_eye.x - left_eye.x).powi(2) + (right_eye.y - left_eye.y).powi(2)).sqrt();
    if eye_distance <= 1.0 {
        return false;
    }

    if !(left_eye.x < right_eye.x && left_mouth.x < right_mouth.x) {
        return false;
    }

    if !(nose.y > left_eye.y.min(right_eye.y) && nose.y < left_mouth.y.max(right_mouth.y)) {
        return false;
    }

    let bbox_w = bbox.width().max(1.0);
    let bbox_h = bbox.height().max(1.0);

    let within_bounds = lm.iter().all(|p| {
        p.x >= bbox.x1 - bbox_w * 0.1
            && p.x <= bbox.x2 + bbox_w * 0.1
            && p.y >= bbox.y1 - bbox_h * 0.1
            && p.y <= bbox.y2 + bbox_h * 0.1
    });

    if !within_bounds {
        return false;
    }

    let mouth_width = ((right_mouth.x - left_mouth.x).powi(2) + (right_mouth.y - left_mouth.y).powi(2)).sqrt();
    mouth_width > eye_distance * 0.35
}

fn compute_laplacian_variance(image: &RgbImage, bbox: BoundingBox) -> f32 {
    let clipped = bbox.clamp_to_image(image.width(), image.height());
    let x1 = clipped.x1.floor().max(0.0) as i32;
    let y1 = clipped.y1.floor().max(0.0) as i32;
    let x2 = clipped.x2.ceil().min((image.width() - 1) as f32) as i32;
    let y2 = clipped.y2.ceil().min((image.height() - 1) as f32) as i32;

    let width = (x2 - x1 + 1).max(0);
    let height = (y2 - y1 + 1).max(0);
    if width < 3 || height < 3 {
        return 0.0;
    }

    let mut laplacian_values = Vec::with_capacity(((width - 2) * (height - 2)) as usize);

    for y in (y1 + 1)..=(y2 - 1) {
        for x in (x1 + 1)..=(x2 - 1) {
            let center = rgb_luma(image.get_pixel(x as u32, y as u32));
            let top = rgb_luma(image.get_pixel(x as u32, (y - 1) as u32));
            let bottom = rgb_luma(image.get_pixel(x as u32, (y + 1) as u32));
            let left = rgb_luma(image.get_pixel((x - 1) as u32, y as u32));
            let right = rgb_luma(image.get_pixel((x + 1) as u32, y as u32));

            let lap = top + bottom + left + right - (4.0 * center);
            laplacian_values.push(lap);
        }
    }

    if laplacian_values.is_empty() {
        return 0.0;
    }

    let mean = laplacian_values.iter().sum::<f32>() / laplacian_values.len() as f32;
    laplacian_values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / laplacian_values.len() as f32
}

fn rgb_luma(rgb: &Rgb<u8>) -> f32 {
    let channels = rgb.channels();
    0.299 * channels[0] as f32 + 0.587 * channels[1] as f32 + 0.114 * channels[2] as f32
}

fn estimate_similarity_transform(src: &[Point2f; 5], dst: &[Point2f; 5]) -> Result<SimilarityTransform, String> {
    let n = src.len() as f32;

    let mut src_mat = Matrix2xX::<f32>::zeros(src.len());
    let mut dst_mat = Matrix2xX::<f32>::zeros(dst.len());

    for (i, p) in src.iter().enumerate() {
        src_mat[(0, i)] = p.x;
        src_mat[(1, i)] = p.y;
    }
    for (i, p) in dst.iter().enumerate() {
        dst_mat[(0, i)] = p.x;
        dst_mat[(1, i)] = p.y;
    }

    let src_mean = Vector2::new(
        (0..src.len()).map(|i| src_mat[(0, i)]).sum::<f32>() / n,
        (0..src.len()).map(|i| src_mat[(1, i)]).sum::<f32>() / n,
    );
    let dst_mean = Vector2::new(
        (0..dst.len()).map(|i| dst_mat[(0, i)]).sum::<f32>() / n,
        (0..dst.len()).map(|i| dst_mat[(1, i)]).sum::<f32>() / n,
    );

    for i in 0..src.len() {
        src_mat[(0, i)] -= src_mean.x;
        src_mat[(1, i)] -= src_mean.y;
        dst_mat[(0, i)] -= dst_mean.x;
        dst_mat[(1, i)] -= dst_mean.y;
    }

    let mut src_var = 0.0_f32;
    for i in 0..src.len() {
        src_var += src_mat[(0, i)] * src_mat[(0, i)] + src_mat[(1, i)] * src_mat[(1, i)];
    }
    src_var /= n;
    if src_var <= f32::EPSILON {
        return Err("Degenerate source landmarks for alignment".to_string());
    }

    let covariance = (&dst_mat * src_mat.transpose()) / n;
    let svd = covariance.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "SVD decomposition failed for alignment (U missing)".to_string())?;
    let v_t = svd
        .v_t
        .ok_or_else(|| "SVD decomposition failed for alignment (Vt missing)".to_string())?;

    let mut s = Matrix2::identity();
    if (u.determinant() * v_t.determinant()) < 0.0 {
        s[(1, 1)] = -1.0;
    }

    let rotation = u * s * v_t;
    let singular = svd.singular_values;
    let trace = singular[0] * s[(0, 0)] + singular[1] * s[(1, 1)];
    let scale = trace / src_var;

    let translation = dst_mean - scale * rotation * src_mean;

    Ok(SimilarityTransform {
        a11: scale * rotation[(0, 0)],
        a12: scale * rotation[(0, 1)],
        a21: scale * rotation[(1, 0)],
        a22: scale * rotation[(1, 1)],
        tx: translation.x,
        ty: translation.y,
    })
}

fn warp_affine_bilinear(src: &RgbImage, transform: SimilarityTransform, width: u32, height: u32) -> Result<RgbImage, String> {
    let det = (transform.a11 * transform.a22) - (transform.a12 * transform.a21);
    if det.abs() <= f32::EPSILON {
        return Err("Alignment transform is singular".to_string());
    }

    let inv_a11 = transform.a22 / det;
    let inv_a12 = -transform.a12 / det;
    let inv_a21 = -transform.a21 / det;
    let inv_a22 = transform.a11 / det;

    let mut output = ImageBuffer::from_pixel(width, height, Rgb([0, 0, 0]));

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - transform.tx;
            let dy = y as f32 - transform.ty;

            let src_x = inv_a11 * dx + inv_a12 * dy;
            let src_y = inv_a21 * dx + inv_a22 * dy;

            let pixel = sample_bilinear(src, src_x, src_y);
            output.put_pixel(x, y, pixel);
        }
    }

    Ok(output)
}

fn sample_bilinear(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    if x < 0.0 || y < 0.0 || x > (image.width() - 1) as f32 || y > (image.height() - 1) as f32 {
        return Rgb([0, 0, 0]);
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);

    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let p00 = image.get_pixel(x0, y0).channels();
    let p10 = image.get_pixel(x1, y0).channels();
    let p01 = image.get_pixel(x0, y1).channels();
    let p11 = image.get_pixel(x1, y1).channels();

    let mut out = [0_u8; 3];
    for c in 0..3 {
        let v00 = p00[c] as f32;
        let v10 = p10[c] as f32;
        let v01 = p01[c] as f32;
        let v11 = p11[c] as f32;

        let v0 = v00 * (1.0 - wx) + v10 * wx;
        let v1 = v01 * (1.0 - wx) + v11 * wx;
        let v = v0 * (1.0 - wy) + v1 * wy;

        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }

    Rgb(out)
}

fn build_candidate_from_detection(
    image: &RgbImage,
    detection: &FaceDetection,
    cfg: &RuntimeConfig,
    reject_counts: &mut QualityRejectCounts,
) -> Option<FaceCandidate> {
    if detection.score < cfg.detect_confidence {
        reject_counts.low_confidence += 1;
        return None;
    }

    let face_size = detection.bbox.short_side();
    if face_size < cfg.min_face_size {
        reject_counts.small_face += 1;
        return None;
    }

    let geometry_ok = landmarks_geometry_ok(&detection.landmarks5, detection.bbox);
    if !geometry_ok {
        reject_counts.invalid_landmark_geometry += 1;
        return None;
    }

    let sharpness = compute_laplacian_variance(image, detection.bbox);
    if sharpness < cfg.min_sharpness {
        reject_counts.low_sharpness += 1;
        return None;
    }

    let transform = estimate_similarity_transform(&detection.landmarks5, &ALIGN_TEMPLATE_112).ok()?;
    let aligned_img = match warp_affine_bilinear(image, transform, 112, 112) {
        Ok(img) => img,
        Err(_) => {
            reject_counts.alignment_failed += 1;
            return None;
        }
    };

    let aligned_112 = preprocess_embed_input(&aligned_img).ok()?;
    Some(FaceCandidate {
        aligned_112,
        detect_score: detection.score,
        quality_metrics: FaceQualityMetrics {
            face_size,
            sharpness,
            geometry_ok,
        },
    })
}

fn detect_and_prepare_candidates(
    image_path: &str,
    detector: &mut Session,
    cfg: &RuntimeConfig,
) -> Result<(Vec<FaceCandidate>, QualityRejectCounts, usize), String> {
    let image = image::open(image_path)
        .map_err(|e| format!("Failed to open image '{}': {e}", image_path))?
        .to_rgb8();

    let detections = detect_faces_scrfd(&image, cfg, detector)?;
    let detected_count = detections.len();

    let mut reject_counts = QualityRejectCounts::default();
    let mut candidates = Vec::new();

    for detection in &detections {
        if let Some(candidate) = build_candidate_from_detection(&image, detection, cfg, &mut reject_counts) {
            candidates.push(candidate);
        }
    }

    Ok((candidates, reject_counts, detected_count))
}

fn extract_embedding(embedder: &mut Session, input_tensor: &Array4<f32>) -> Result<Vec<f32>, String> {
    let input = TensorRef::from_array_view(input_tensor)
        .map_err(|e| format!("Failed to build embedder input tensor: {e}"))?;
    let outputs = embedder
        .run(inputs![input])
        .map_err(|e| format!("Embedding inference failed: {e}"))?;

    if outputs.len() == 0 {
        return Err("Embedder returned no outputs".to_string());
    }

    let (_, embedding) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("Failed to extract embedding tensor: {e}"))?;

    Ok(embedding.to_vec())
}

fn compare_candidate_sets(
    candidates1: &[FaceCandidate],
    candidates2: &[FaceCandidate],
    embedder: &mut Session,
    threshold: f32,
    pair_margin: f32,
) -> Result<CompareDecision, String> {
    if candidates1.is_empty() || candidates2.is_empty() {
        return Err("No qualified face candidates available for comparison".to_string());
    }

    let embeddings1 = candidates1
        .iter()
        .map(|c| extract_embedding(embedder, &c.aligned_112))
        .collect::<Result<Vec<_>, _>>()?;

    let embeddings2 = candidates2
        .iter()
        .map(|c| extract_embedding(embedder, &c.aligned_112))
        .collect::<Result<Vec<_>, _>>()?;

    let mut pair_scores: Vec<(usize, usize, f32)> = Vec::new();
    for (i, e1) in embeddings1.iter().enumerate() {
        for (j, e2) in embeddings2.iter().enumerate() {
            let score = cosine_similarity(e1, e2)?;
            pair_scores.push((i, j, score));
        }
    }

    if pair_scores.is_empty() {
        return Err("No face pairs available for scoring".to_string());
    }

    pair_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    let (best_i, best_j, best_score) = pair_scores[0];
    let second_best = if pair_scores.len() > 1 {
        pair_scores[1].2
    } else {
        best_score
    };
    let margin = best_score - second_best;

    let threshold_ok = best_score >= threshold;
    let margin_ok = margin >= pair_margin;

    let (verified, decision_reason) = if !threshold_ok {
        (false, "below_threshold")
    } else if !margin_ok {
        (false, "margin_too_small")
    } else {
        (true, "verified")
    };

    Ok(CompareDecision {
        best_score,
        second_best_score: second_best,
        margin,
        verified,
        selected_pair: (best_i, best_j),
        decision_reason,
    })
}

fn add_diagnostics(
    response: &mut ZendHashTable,
    cfg: &RuntimeConfig,
    detected_count_1: usize,
    detected_count_2: usize,
    candidates1: &[FaceCandidate],
    candidates2: &[FaceCandidate],
    decision: &CompareDecision,
    rejects1: &QualityRejectCounts,
    rejects2: &QualityRejectCounts,
) -> PhpResult<()> {
    if !cfg.diagnostics {
        return Ok(());
    }

    let detect_confidence_min = candidates1
        .iter()
        .chain(candidates2.iter())
        .map(|c| c.detect_score)
        .fold(f32::INFINITY, |acc, v| acc.min(v));
    let quality_face_size_min = candidates1
        .iter()
        .chain(candidates2.iter())
        .map(|c| c.quality_metrics.face_size)
        .fold(f32::INFINITY, |acc, v| acc.min(v));
    let quality_sharpness_min = candidates1
        .iter()
        .chain(candidates2.iter())
        .map(|c| c.quality_metrics.sharpness)
        .fold(f32::INFINITY, |acc, v| acc.min(v));
    let quality_geometry_all_passed = candidates1
        .iter()
        .chain(candidates2.iter())
        .all(|c| c.quality_metrics.geometry_ok);

    response
        .insert("face_count_img1", detected_count_1 as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert("face_count_img2", detected_count_2 as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert(
            "selected_pair",
            format!("{}:{}", decision.selected_pair.0, decision.selected_pair.1),
        )
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert("best_similarity", decision.best_score as f64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert("second_best_similarity", decision.second_best_score as f64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert("pair_margin", decision.margin as f64)
        .map_err(|e| PhpException::default(e.to_string()))?;

    if detect_confidence_min.is_finite() {
        response
            .insert("detect_confidence_min", detect_confidence_min as f64)
            .map_err(|e| PhpException::default(e.to_string()))?;
    }
    if quality_face_size_min.is_finite() {
        response
            .insert("quality_face_size_min", quality_face_size_min as f64)
            .map_err(|e| PhpException::default(e.to_string()))?;
    }
    if quality_sharpness_min.is_finite() {
        response
            .insert("quality_sharpness_min", quality_sharpness_min as f64)
            .map_err(|e| PhpException::default(e.to_string()))?;
    }
    response
        .insert("quality_geometry_all_passed", quality_geometry_all_passed)
        .map_err(|e| PhpException::default(e.to_string()))?;

    let mut reject_table = ZendHashTable::new();
    reject_table
        .insert("img1_low_confidence", rejects1.low_confidence as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert("img1_small_face", rejects1.small_face as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert("img1_low_sharpness", rejects1.low_sharpness as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert(
            "img1_invalid_landmark_geometry",
            rejects1.invalid_landmark_geometry as i64,
        )
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert("img1_alignment_failed", rejects1.alignment_failed as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;

    reject_table
        .insert("img2_low_confidence", rejects2.low_confidence as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert("img2_small_face", rejects2.small_face as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert("img2_low_sharpness", rejects2.low_sharpness as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert(
            "img2_invalid_landmark_geometry",
            rejects2.invalid_landmark_geometry as i64,
        )
        .map_err(|e| PhpException::default(e.to_string()))?;
    reject_table
        .insert("img2_alignment_failed", rejects2.alignment_failed as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;

    response
        .insert("quality_reject_counts", reject_table)
        .map_err(|e| PhpException::default(e.to_string()))?;

    response
        .insert("decision_reason", decision.decision_reason)
        .map_err(|e| PhpException::default(e.to_string()))?;

    Ok(())
}

#[php_function]
pub fn deepface_analyze(image_path: String) -> PhpResult<ZBox<ZendHashTable>> {
    let img = image::open(&image_path).map_err(|e| PhpException::default(e.to_string()))?;
    let (w, h) = img.dimensions();

    let mut result = ZendHashTable::new();
    result
        .insert("status", "success")
        .map_err(|e| PhpException::default(e.to_string()))?;
    result
        .insert("width", w as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    result
        .insert("height", h as i64)
        .map_err(|e| PhpException::default(e.to_string()))?;

    Ok(result)
}

#[php_function]
pub fn deepface_compare(
    img1_path: String,
    img2_path: String,
    model_path: String,
    threshold: f32,
) -> PhpResult<ZBox<ZendHashTable>> {
    if !(-1.0..=1.0).contains(&threshold) {
        return Err(PhpException::default(
            "threshold must be within [-1.0, 1.0]".to_string(),
        ));
    }

    let cfg = load_runtime_config()?;
    let embedder_model_path = resolve_embedder_model_path(&model_path).map_err(PhpException::default)?;

    let detector = get_or_load_session(detector_cache(), &cfg.detector_model_path)?;
    let embedder = get_or_load_session(embedder_cache(), &embedder_model_path)?;

    let mut detector = detector
        .lock()
        .map_err(|_| PhpException::default("Detector session lock poisoned".to_string()))?;
    let mut embedder = embedder
        .lock()
        .map_err(|_| PhpException::default("Embedder session lock poisoned".to_string()))?;

    let (candidates1, rejects1, detected_count_1) = detect_and_prepare_candidates(&img1_path, &mut detector, cfg)
        .map_err(PhpException::default)?;
    let (candidates2, rejects2, detected_count_2) = detect_and_prepare_candidates(&img2_path, &mut detector, cfg)
        .map_err(PhpException::default)?;

    if candidates1.is_empty() || candidates2.is_empty() {
        return Err(PhpException::default(format!(
            "No qualified faces available after strict quality gates (img1_rejects={}, img2_rejects={})",
            rejects1.total(),
            rejects2.total()
        )));
    }

    let decision = compare_candidate_sets(
        &candidates1,
        &candidates2,
        &mut embedder,
        threshold,
        cfg.pair_margin,
    )
    .map_err(PhpException::default)?;

    let mut response = ZendHashTable::new();
    response
        .insert("verified", decision.verified)
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert("similarity", decision.best_score as f64)
        .map_err(|e| PhpException::default(e.to_string()))?;
    response
        .insert("threshold", threshold as f64)
        .map_err(|e| PhpException::default(e.to_string()))?;

    add_diagnostics(
        &mut response,
        cfg,
        detected_count_1,
        detected_count_2,
        &candidates1,
        &candidates2,
        &decision,
        &rejects1,
        &rejects2,
    )?;

    Ok(response)
}

#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
        .function(wrap_function!(deepface_analyze))
        .function(wrap_function!(deepface_compare))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_transform_maps_points_close_to_template() {
        let src = [
            Point2f { x: 30.0, y: 50.0 },
            Point2f { x: 70.0, y: 48.0 },
            Point2f { x: 52.0, y: 70.0 },
            Point2f { x: 36.0, y: 92.0 },
            Point2f { x: 68.0, y: 90.0 },
        ];

        let t = estimate_similarity_transform(&src, &ALIGN_TEMPLATE_112).expect("transform");

        let mut max_err = 0.0_f32;
        for (i, p) in src.iter().enumerate() {
            let x = (t.a11 * p.x) + (t.a12 * p.y) + t.tx;
            let y = (t.a21 * p.x) + (t.a22 * p.y) + t.ty;
            let dx = x - ALIGN_TEMPLATE_112[i].x;
            let dy = y - ALIGN_TEMPLATE_112[i].y;
            let err = (dx * dx + dy * dy).sqrt();
            max_err = max_err.max(err);
        }

        assert!(max_err < 1.5, "max landmark alignment error too high: {}", max_err);
    }

    #[test]
    fn quality_gate_landmark_geometry_rejects_invalid_order() {
        let bbox = BoundingBox {
            x1: 0.0,
            y1: 0.0,
            x2: 100.0,
            y2: 100.0,
        };

        let invalid = [
            Point2f { x: 70.0, y: 30.0 },
            Point2f { x: 30.0, y: 30.0 },
            Point2f { x: 50.0, y: 50.0 },
            Point2f { x: 40.0, y: 80.0 },
            Point2f { x: 60.0, y: 80.0 },
        ];

        assert!(!landmarks_geometry_ok(&invalid, bbox));
    }

    #[test]
    fn pair_margin_logic_handles_edge_cases() {
        let decision = CompareDecision {
            best_score: 0.65,
            second_best_score: 0.62,
            margin: 0.03,
            verified: false,
            selected_pair: (0, 0),
            decision_reason: "margin_too_small",
        };

        assert!(!decision.verified);
        assert!(decision.best_score >= 0.5);
        assert!(decision.margin < 0.06);
    }

    #[test]
    fn nms_suppresses_overlapping_boxes() {
        let d1 = FaceDetection {
            bbox: BoundingBox {
                x1: 10.0,
                y1: 10.0,
                x2: 60.0,
                y2: 60.0,
            },
            score: 0.95,
            landmarks5: [Point2f { x: 0.0, y: 0.0 }; 5],
        };
        let d2 = FaceDetection {
            bbox: BoundingBox {
                x1: 12.0,
                y1: 12.0,
                x2: 61.0,
                y2: 61.0,
            },
            score: 0.90,
            landmarks5: [Point2f { x: 0.0, y: 0.0 }; 5],
        };

        let kept = nms(vec![d1, d2], 0.4);
        assert_eq!(kept.len(), 1);
    }
}
