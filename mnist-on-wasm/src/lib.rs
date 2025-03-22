use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, Worker, console};

mod mnist_model;
use mnist_model::MnistModel;

#[wasm_bindgen]
pub struct MNIST {
    canvas_data: Vec<u8>,
    model: Option<MnistModel>,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl MNIST {
    pub fn new() -> MNIST {
        MNIST {
            canvas_data: Vec::new(),
            model: None,
            width: 28,
            height: 28,
        }
    }

    pub fn execute(
        &mut self,
        canvas_data: Vec<u8>,
        width: u32,
        height: u32,
    ) -> Result<usize, JsValue> {
        self.canvas_data = canvas_data;
        self.width = width;
        self.height = height;

        // 必要に応じてモデルを初期化
        if self.model.is_none() {
            self.model = Some(MnistModel::new()?);
        }

        // キャンバスデータから推論実行
        if let Some(model) = &self.model {
            let result = model.process_image(&self.canvas_data, self.width, self.height)?;
            Ok(result)
        } else {
            Err(JsValue::from_str("モデルが初期化されていません"))
        }
    }
}

/// Run entry point for the main thread.
#[wasm_bindgen]
pub fn startup() {
    let _worker_handle = Rc::new(RefCell::new(Worker::new("./assets/worker.js").unwrap()));
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("canvas").unwrap();
    let canvas: HtmlCanvasElement = canvas
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    // キャンバスのサイズを設定
    canvas.set_width(280);
    canvas.set_height(280);

    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();

    // 背景を白で塗る
    context.set_fill_style(&JsValue::from_str("#FFFFFF").into());
    context.set_global_alpha(1.0);
    context.fill_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);

    context.set_line_width(20.0);
    context.set_line_cap("round");
    context.set_stroke_style(&JsValue::from_str("#000000").into());

    // 描画状態を管理
    let is_drawing = Rc::new(RefCell::new(false));
    let last_x = Rc::new(RefCell::new(0.0));
    let last_y = Rc::new(RefCell::new(0.0));

    // MNISTモデルのインスタンス作成
    let mnist = Rc::new(RefCell::new(MNIST::new()));

    // マウスダウンイベント
    let mouse_down_callback = Closure::wrap(Box::new({
        let is_drawing = is_drawing.clone();
        let last_x = last_x.clone();
        let last_y = last_y.clone();

        move |event: web_sys::Event| {
            *is_drawing.borrow_mut() = true;

            // EventをMouseEventとして扱うことはできないので、offsetXとoffsetYを取得するためにJSの方法を使う
            let event_js = event.dyn_ref::<js_sys::Object>().unwrap();
            let x = js_sys::Reflect::get(event_js, &JsValue::from_str("offsetX"))
                .unwrap()
                .as_f64()
                .unwrap();
            let y = js_sys::Reflect::get(event_js, &JsValue::from_str("offsetY"))
                .unwrap()
                .as_f64()
                .unwrap();

            *last_x.borrow_mut() = x;
            *last_y.borrow_mut() = y;
        }
    }) as Box<dyn FnMut(_)>);

    canvas
        .add_event_listener_with_callback("mousedown", mouse_down_callback.as_ref().unchecked_ref())
        .unwrap();
    mouse_down_callback.forget();

    // マウス移動イベント
    let mouse_move_callback = Closure::wrap(Box::new({
        let is_drawing = is_drawing.clone();
        let last_x = last_x.clone();
        let last_y = last_y.clone();
        let context = context.clone();

        move |event: web_sys::Event| {
            if !*is_drawing.borrow() {
                return;
            }

            // EventをMouseEventとして扱うことはできないので、offsetXとoffsetYを取得するためにJSの方法を使う
            let event_js = event.dyn_ref::<js_sys::Object>().unwrap();
            let current_x = js_sys::Reflect::get(event_js, &JsValue::from_str("offsetX"))
                .unwrap()
                .as_f64()
                .unwrap();
            let current_y = js_sys::Reflect::get(event_js, &JsValue::from_str("offsetY"))
                .unwrap()
                .as_f64()
                .unwrap();

            context.begin_path();
            context.move_to(*last_x.borrow(), *last_y.borrow());
            context.line_to(current_x, current_y);
            context.stroke();

            *last_x.borrow_mut() = current_x;
            *last_y.borrow_mut() = current_y;
        }
    }) as Box<dyn FnMut(_)>);

    canvas
        .add_event_listener_with_callback("mousemove", mouse_move_callback.as_ref().unchecked_ref())
        .unwrap();
    mouse_move_callback.forget();

    // マウスアップイベント
    let mouse_up_callback = Closure::wrap(Box::new({
        let is_drawing = is_drawing.clone();

        move |_event: web_sys::Event| {
            *is_drawing.borrow_mut() = false;
        }
    }) as Box<dyn FnMut(_)>);

    canvas
        .add_event_listener_with_callback("mouseup", mouse_up_callback.as_ref().unchecked_ref())
        .unwrap();
    mouse_up_callback.forget();

    // クリアボタン
    let clear_button = document
        .get_element_by_id("clear-button")
        .expect("クリアボタンが見つかりません");
    let clear_callback = Closure::wrap(Box::new({
        let context = context.clone();
    let canvas = canvas.clone();
    let mnist = mnist.clone();
    let result = document.get_element_by_id("result").unwrap();

    move |_event: web_sys::Event| {
        // キャンバスをクリア
        context.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);

        // 背景を白で塗り直す
        context.set_fill_style(&JsValue::from_str("#FFFFFF").into());
        context.fill_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);

        // `MNIST` インスタンスのデータをリセット
        mnist.borrow_mut().canvas_data.clear();

        // 認識結果をクリア
        result.set_text_content(Some(&format!("")));
    }
    }) as Box<dyn FnMut(_)>);

    clear_button
        .add_event_listener_with_callback("click", clear_callback.as_ref().unchecked_ref())
        .unwrap();
    clear_callback.forget();

    // 認識ボタン
    let recognize_button = document
        .get_element_by_id("recognize-button")
        .expect("認識ボタンが見つかりません");
    let recognize_callback = Closure::wrap(Box::new({
        let canvas = canvas.clone();
        let mnist = mnist.clone();
        let context = context.clone();

        move |_event: web_sys::Event| {
            // キャンバスから画像データを取得
            let width = canvas.width();
            let height = canvas.height();

            let image_data = context
                .get_image_data(0.0, 0.0, width as f64, height as f64)
                .unwrap_or_else(|_| panic!("キャンバスデータの取得に失敗"));

            let pixels = image_data.data();
            let data_vec = pixels.to_vec();
            
            // MNISTモデルに渡して実行
            match mnist.borrow_mut().execute(data_vec, width, height) {
                Ok(digit) => {
                    if let Some(element) = document.get_element_by_id("result") {
                        element.set_text_content(Some(&format!("認識結果: {}", digit)));
                    }
                }
                Err(e) => {
                    console::error_1(&format!("認識処理エラー: {:?}", e).into());
                }
            }
        }
    }) as Box<dyn FnMut(_)>);

    recognize_button
        .add_event_listener_with_callback("click", recognize_callback.as_ref().unchecked_ref())
        .unwrap();
    recognize_callback.forget();
}
