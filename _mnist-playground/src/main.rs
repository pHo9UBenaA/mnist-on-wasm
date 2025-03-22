use image::imageops;
use ndarray::ArrayD;
use ort::session::{Session, SessionOutputs, builder::GraphOptimizationLevel};
use std::path::Path;

fn load_model() -> Result<Session, ort::Error> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file("assets/mnist-12.onnx")?;
    Ok(model)
}

fn image_to_array() -> Result<ArrayD<f32>, ort::Error> {
    let img = image::open(Path::new("assets/2.png")).expect("Failed to open image");

    // グレースケールに変換
    let gray = img.to_luma8();

    // 28x28にリサイズ
    let resized = imageops::resize(&gray, 28, 28, imageops::FilterType::Lanczos3);

    let flat_img: Vec<f32> = resized.pixels().map(|p| p[0] as f32 / 255.0).collect();

    let image = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 1, 28, 28]), flat_img)
        .expect("画像データの形状変換に失敗しました");

    Ok(image)
}

fn process_result(outputs: &SessionOutputs<'_, '_>) -> Result<usize, ort::Error> {
    let predictions = outputs["Plus214_Output_0"].try_extract_tensor::<f32>()?;

    let max_index = predictions
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .expect("Failed to find max value");

    Ok(max_index)
}

fn main() -> Result<(), ort::Error> {
    let model = load_model()?;

    let image = image_to_array()?;

    let outputs = model.run(ort::inputs!["Input3" => image]?)?;

    let result = process_result(&outputs)?;

    println!("Predicted digit: {}", result);

    Ok(())
}
