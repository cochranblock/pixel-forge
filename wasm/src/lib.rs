// Unlicense — cochranblock.org
//! Pixel Forge browser runtime.
//!
//! Exposes the Cinder TinyUNet diffusion model to the dndaimodel.org page
//! via WebGPU (wgpu → any-gpu). Forward inference only; training stays on
//! the desktop binary.
//!
//! Public JS surface:
//!   - `init_panic_hook()`     — wire console.error backtraces
//!   - `webgpu_available()`    — pre-flight, no GPU touch
//!   - `boot()`                — async device init, returns adapter name
//!   - `generate(...)`         — full sprite gen (added in later task)

use wasm_bindgen::prelude::*;

// Reuse the canonical class conditioning table from the desktop crate. Pure
// Rust, zero deps — including via #[path] keeps a single source of truth.
#[path = "../../src/class_cond.rs"]
mod class_cond;

mod loader;
mod timestep;
mod weights;
mod tiny_unet;
mod sampler;
mod normalize;

use std::cell::RefCell;

mod log {
    use wasm_bindgen::prelude::*;
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        pub fn log(s: &str);
        #[wasm_bindgen(js_namespace = console)]
        pub fn error(s: &str);
    }
}

/// Install a panic hook so Rust panics surface as `console.error` with a
/// readable backtrace. Cheap; idempotent. Call once on page load.
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

/// Quick feature-detect — `navigator.gpu !== undefined`. Doesn't request an
/// adapter; safe to call before any heavy code paths.
#[wasm_bindgen]
pub fn webgpu_available() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("navigator"))
        .ok()
        .and_then(|nav| js_sys::Reflect::get(&nav, &JsValue::from_str("gpu")).ok())
        .map(|gpu| !gpu.is_undefined() && !gpu.is_null())
        .unwrap_or(false)
}

/// Async runtime boot. Initializes any-gpu's WebGPU device, runs a 4-element
/// add as a smoke test, and returns the adapter name. Throws (Promise reject)
/// with a readable error string on failure.
#[wasm_bindgen]
pub async fn boot() -> Result<String, JsValue> {
    let dev = any_gpu::GpuDevice::gpu_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("device init: {e:#}")))?;

    let a = dev.upload(&[1.0, 2.0, 3.0, 4.0]);
    let b = dev.upload(&[10.0, 20.0, 30.0, 40.0]);
    let sum = dev
        .add(&a, &b)
        .map_err(|e| JsValue::from_str(&format!("add: {e:#}")))?;
    let result = dev
        .read_async(&sum)
        .await
        .map_err(|e| JsValue::from_str(&format!("readback: {e:#}")))?;

    let want = [11.0, 22.0, 33.0, 44.0];
    if result != want {
        return Err(JsValue::from_str(&format!(
            "smoke test wrong: got {result:?} want {want:?}"
        )));
    }

    let banner = format!("{} via {}", dev.adapter_name, dev.backend);
    DEVICE.with(|d| *d.borrow_mut() = Some(dev));
    log::log(&format!("pixel-forge-wasm boot: {banner}"));
    Ok(banner)
}

// Per-page singleton state. wasm32-unknown-unknown is single-threaded so a
// thread-local + RefCell is enough — no Mutex needed.
thread_local! {
    static DEVICE: RefCell<Option<any_gpu::GpuDevice>> = const { RefCell::new(None) };
    static MODEL: RefCell<Option<weights::CinderW>> = const { RefCell::new(None) };
    static NORMALIZER: RefCell<Option<normalize::Normalizer>> = const { RefCell::new(None) };
}

/// Install the per-channel z-score that travels with this checkpoint. JS
/// fetches the `.normalize.json` sidecar (404 = no normalizer; do not
/// call) and passes the raw bytes here. Subsequent `generate` calls apply
/// the inverse transform before clamp + PNG encode.
#[wasm_bindgen]
pub fn set_normalizer(bytes: &[u8]) -> Result<(), JsValue> {
    let n = normalize::Normalizer::from_json_bytes(bytes)
        .map_err(|e| JsValue::from_str(&format!("normalize manifest: {e:#}")))?;
    log::log(&format!(
        "pixel-forge-wasm: z-score loaded, mean={:?} std={:?}",
        n.mean, n.std
    ));
    NORMALIZER.with(|c| *c.borrow_mut() = Some(n));
    Ok(())
}

/// Upload the Cinder model. JS fetches the safetensors blob (PNG-style:
/// fetch → arrayBuffer → Uint8Array) and passes it here once. Subsequent
/// `generate` calls reuse the GPU-resident weights.
///
/// Returns the parameter count for UI display.
#[wasm_bindgen]
pub fn load_model(bytes: &[u8]) -> Result<u32, JsValue> {
    let tensors = loader::load(bytes)
        .map_err(|e| JsValue::from_str(&format!("safetensors: {e:#}")))?;
    let total: usize = tensors.iter().map(|t| t.data.len()).sum();

    let cinder = DEVICE.with(|d| {
        let d = d.borrow();
        let dev = d.as_ref().ok_or_else(|| {
            JsValue::from_str("device not initialized — call boot() first")
        })?;
        weights::CinderW::load(dev, &tensors)
            .map_err(|e| JsValue::from_str(&format!("weights: {e:#}")))
    })?;

    MODEL.with(|m| *m.borrow_mut() = Some(cinder));
    Ok(total as u32)
}

/// Generate one 32×32 sprite. Returns PNG-encoded RGBA bytes.
///
/// - `class_name`: a key into `class_cond::lookup` ("character", "dragon", etc.)
/// - `seed`: u32 reproducibility seed.
/// - `steps`: ~40 is the desktop default; lower is faster but blurrier.
#[wasm_bindgen]
pub async fn generate(class_name: &str, seed: u32, steps: u32) -> Result<Vec<u8>, JsValue> {
    let cond = class_cond::lookup(class_name);
    let img_size: u32 = 32;

    // Borrow Device + Model out into local references that live for the
    // duration of the async sample. We need to release the RefCell borrow
    // before awaiting (RefCell can't cross await points).
    let (dev, model) = take_dev_and_model()?;

    let result = sampler::sample(
        &dev,
        &model,
        cond.super_id,
        &cond.tags,
        img_size,
        steps as usize,
        seed as u64,
    )
    .await
    .map_err(|e| JsValue::from_str(&format!("sample: {e:#}")))?;

    // Snapshot the optional normalizer for the lifetime of finalize_png.
    let nrm = NORMALIZER.with(|c| c.borrow().clone());
    let png = sampler::finalize_png(&dev, &result, img_size, nrm.as_ref())
        .await
        .map_err(|e| JsValue::from_str(&format!("png: {e:#}")))?;

    // Return state to the cells.
    DEVICE.with(|d| *d.borrow_mut() = Some(dev));
    MODEL.with(|m| *m.borrow_mut() = Some(model));

    Ok(png)
}

fn take_dev_and_model() -> Result<(any_gpu::GpuDevice, weights::CinderW), JsValue> {
    let dev = DEVICE
        .with(|d| d.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("device not initialized — call boot() first"))?;
    let model = MODEL
        .with(|m| m.borrow_mut().take())
        .ok_or_else(|| JsValue::from_str("model not loaded — call load_model() first"))?;
    Ok((dev, model))
}
