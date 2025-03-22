use image::{imageops, ImageBuffer, Luma};
use tract_onnx::prelude::*;
use wasm_bindgen::prelude::*;

pub struct MnistModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl MnistModel {
    pub fn new() -> Result<Self, JsValue> {
        // モデルの読み込み（バイナリデータとしてコンパイル時に埋め込む）
        let model_bytes = include_bytes!("../assets/mnist-12.onnx");

        // tract-onnxでモデルをロード
        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))
            .map_err(|e| JsValue::from_str(&format!("モデルのロードに失敗: {}", e)))?
            .into_optimized()
            .map_err(|e| JsValue::from_str(&format!("モデルの最適化に失敗: {}", e)))?
            .into_runnable()
            .map_err(|e| JsValue::from_str(&format!("実行可能なモデルへの変換に失敗: {}", e)))?;

        Ok(MnistModel { model })
    }

    pub fn process_image(
        &self,
        canvas_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<usize, JsValue> {
        // RGBA形式かどうかチェック (各ピクセルは4バイト)
        if canvas_data.len() != (width * height * 4) as usize {
            return Err(JsValue::from_str(&format!(
                "キャンバスデータのサイズが不正: {}x{}x4={} ≠ {}",
                width,
                height,
                width * height * 4,
                canvas_data.len()
            )));
        }

        // RGBAデータからグレースケールを作成 - アルファチャンネルも考慮
        let mut gray_data = vec![0u8; (width * height) as usize];
        for (i, chunk) in canvas_data.chunks(4).enumerate() {
            if i < gray_data.len() && chunk.len() >= 4 {
                // アルファ値を考慮した計算
                let alpha = chunk[3] as f32 / 255.0;
                let r = (chunk[0] as f32) * alpha;
                let g = (chunk[1] as f32) * alpha;
                let b = (chunk[2] as f32) * alpha;

                let gray_value = (r + g + b) / 3.0;
                gray_data[i] = gray_value as u8;
            }
        }

        // グレースケールデータからImageBufferを作成
        let img_buffer = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, gray_data)
            .ok_or_else(|| JsValue::from_str("画像バッファの作成に失敗"))?;

        // 前処理して推論用テンソルを作成
        let input_tensor = self.preprocess_image(img_buffer)?;

        // 推論実行
        let outputs = self
            .model
            .run(tvec!(input_tensor.into()))
            .map_err(|e| JsValue::from_str(&format!("推論実行エラー: {}", e)))?;

        // 結果を解析して最も確率の高い数字を返す
        let output = outputs[0]
            .to_array_view::<f32>()
            .map_err(|e| JsValue::from_str(&format!("出力テンソル変換エラー: {}", e)))?;

        // 最大値のインデックスを見つける
        let mut max_index = 0;
        let mut max_value = output[[0, 0]];

        for i in 1..10 {
            if output[[0, i]] > max_value {
                max_value = output[[0, i]];
                max_index = i;
            }
        }

        Ok(max_index)
    }

    // 画像の前処理: img_bufferはすでにグレースケール化されているが、280*280にリサイズする
    fn preprocess_image(
        &self,
        img_buffer: ImageBuffer<Luma<u8>, Vec<u8>>,
    ) -> Result<Tensor, JsValue> {
        // 28x28にリサイズ
        let resized_data = imageops::resize(&img_buffer, 28, 28, imageops::FilterType::Lanczos3);

        // リサイズ後の画像の内容を確認
        let max_resized = resized_data.pixels().map(|p| p[0]).max().unwrap_or(0);
        let min_resized = resized_data.pixels().map(|p| p[0]).min().unwrap_or(0);

        // しきい値を動的に設定（最大値と最小値の中間）
        let threshold = if max_resized > min_resized {
            min_resized + (max_resized - min_resized) / 2
        } else {
            50 // デフォルト値
        };

        let mut enhanced_data = vec![0u8; resized_data.len()];
        for (i, pixel) in resized_data.pixels().enumerate() {
            // しきい値以上の値を強調（コントラスト強調）
            if pixel[0] > threshold {
                enhanced_data[i] = 255; // 値のあるピクセルは最大値に
            } else {
                enhanced_data[i] = 0; // ノイズは0に
            }
        }

        // 強調後の画像の内容を確認
        let non_zero_enhanced = enhanced_data.iter().filter(|&&p| p > 0).count();

        // 空のキャンバスの場合はエラーを返す
        if non_zero_enhanced == 0 {
            return Err(JsValue::from_str(
                "何も描かれていません。数字を描いてから認識ボタンを押してください。",
            ));
        }

        // グレースケール値を0-1の範囲に正規化し、白黒反転（MNISTは黒背景に白文字）
        let mut normalized_data = vec![0.0f32; enhanced_data.len()];
        for (i, &pixel) in enhanced_data.iter().enumerate() {
            // MNISTは白黒反転させる（黒背景に白文字）
            normalized_data[i] = 1.0 - (pixel as f32 / 255.0);
        }

        // MNISTモデルの入力形式にテンソルを整形 [1, 1, 28, 28]
        let tensor = tract_ndarray::Array4::from_shape_fn((1, 1, 28, 28), |(_, _, y, x)| {
            let index = y * 28 + x;
            if index < normalized_data.len() {
                normalized_data[index]
            } else {
                0.0
            }
        });

        // tract-onnxのテンソル形式に変換
        let tensor = tensor.into_tensor();

        Ok(tensor)
    }
}
