use ext_php_rs::prelude::*;
use ext_php_rs::types::Zarray;
use ort::{inputs, Session};
use ndarray::{Array4, Axis};
use image::{GenericImageView, ImageBuffer, RGB};

// Helper function to preprocess an image into an ONNX-friendly Tensor Array
fn preprocess_face(image_path: &str, target_dim: u32) -> Result<Array4<f32>, String> {
    let img = image::open(image_path).map_err(|e| e.to_string())?;
    let resized = img.resize_exact(target_dim, target_dim, image::imageops::FilterType::Triangle);
    
    // Normalize pixels to [-1, 1] or [0, 1] depending on your ONNX model rules
    let mut array = Array4::zeros((1, 3, target_dim as usize, target_dim as usize));
    for (x, y, pixel) in resized.pixels() {
        let r = (pixel[0] as f32 / 127.5) - 1.0;
        let g = (pixel[1] as f32 / 127.5) - 1.0;
        let b = (pixel[2] as f32 / 127.5) - 1.0;
        
        array[[0, 0, y as usize, x as usize]] = r;
        array[[0, 1, y as usize, x as usize]] = g;
        array[[0, 2, y as usize, x as usize]] = b;
    }
    Ok(array)
}

// Helper function to compute vector similarity
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot_product: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f32 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm_b: f32 = v2.iter().map(|b| b * b).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

/// 1. Analyze a Face 
/// Returns standard dictionary payload of attributes (mocked structure)
#[php_function]
pub fn deepface_analyze(image_path: String) -> PhpResult<Zarray> {
    // In production, load an attribute ONNX model (e.g., gender/age)
    // For this boilerplate, we validate the image parses and output structured metrics
    let img = image::open(&image_path).map_err(|e| PhpException::default(e.to_string()))?;
    let (w, h) = img.dimensions();

    let mut result = Zarray::new();
    result.insert("status", "success")?;
    result.insert("width", w as i64)?;
    result.insert("height", h as i64)?;
    // Add logic here to append model evaluations for age, gender, or emotion
    
    Ok(result)
}

/// 2. Compare Two Faces
/// Extracts 512-D embeddings using ArcFace ONNX model and checks thresholds
#[php_function]
pub fn deepface_compare(img1_path: String, img2_path: String, model_path: String, threshold: f32) -> PhpResult<Zarray> {
    // Load your compiled ONNX model (e.g., arcface_w600k_r50.onnx)
    let model = Session::builder()
        .map_err(|e| PhpException::default(e.to_string()))?
        .commit_from_file(&model_path)
        .map_err(|e| PhpException::default(e.to_string()))?;

    // Preprocess images (ArcFace typical dimension is 112x112)
    let tensor1 = preprocess_face(&img1_path, 112).map_err(|e| PhpException::default(e))?;
    let tensor2 = preprocess_face(&img2_path, 112).map_err(|e| PhpException::default(e))?;

    // Extract Vector Embeddings for Image 1
    let outputs1 = model.run(inputs![tensor1].unwrap()).map_err(|e| PhpException::default(e.to_string()))?;
    let embedding1_tensor = outputs1[0].try_extract_tensor::<f32>().map_err(|e| PhpException::default(e.to_string()))?;
    let embedding1: Vec<f32> = embedding1_tensor.iter().cloned().collect();

    // Extract Vector Embeddings for Image 2
    let outputs2 = model.run(inputs![tensor2].unwrap()).map_err(|e| PhpException::default(e.to_string()))?;
    let embedding2_tensor = outputs2[0].try_extract_tensor::<f32>().map_err(|e| PhpException::default(e.to_string()))?;
    let embedding2: Vec<f32> = embedding2_tensor.iter().cloned().collect();

    // Compute metrics
    let similarity = cosine_similarity(&embedding1, &embedding2);
    let is_match = similarity >= threshold;

    // Build standard DeepFace style output array
    let mut response = Zarray::new();
    response.insert("verified", is_match)?;
    response.insert("similarity", similarity as f64)?;
    response.insert("threshold", threshold as f64)?;

    Ok(response)
}

#[php_module]
pub fn get_module(module: ModuleBuilder) -> ModuleBuilder {
    module
}
