//! Joi Burn v3 — TEMPORAL 300-Brain inference on Metal
//!
//! v3 change: Input is [seq_len=20, 54] temporal sequence.
//! Attention is real Q@K^T (20×20 per head) instead of V-only.
//! Weights are pre-stacked [300, out, in] from training.

use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use safetensors::SafeTensors;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::time::Instant;

const INPUT_DIM: usize = 54;
const SEQ_LEN: usize = 20;
const NUM_BRAINS: usize = 300;
const D_MODEL: usize = 256;
const NUM_LAYERS: usize = 4;
const NUM_HEADS: usize = 8;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS; // 32
const FFN_DIM: usize = 1024;
const NUM_CLASSES: usize = 4;

const CLASS_NAMES: [&str; NUM_CLASSES] = ["OBSERVE", "HOLD", "ENTER_LONG", "ENTER_SHORT"];
const SYMBOL_MAP: [&str; 6] = ["BTC", "ETH", "SOL", "HYPE", "BNB", "LINK"];

// ═══ Weight helpers ════════════════════════════════════════════════

fn load_f32_vec(st: &SafeTensors, key: &str) -> Vec<f32> {
    let view = st
        .tensor(key)
        .unwrap_or_else(|e| panic!("Missing: {key} — {e}"));
    let data = view.data();
    assert!(data.len() % 4 == 0, "Expected FP32 for {key}");
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4) }.to_vec()
}

fn load_t1<B: Backend>(st: &SafeTensors, key: &str, d: usize, dev: &B::Device) -> Tensor<B, 1> {
    let v = load_f32_vec(st, key);
    assert_eq!(v.len(), d, "{key}: expected {d}, got {}", v.len());
    Tensor::<B, 1>::from_floats(v.as_slice(), dev)
}

fn load_t2<B: Backend>(
    st: &SafeTensors,
    key: &str,
    r: usize,
    c: usize,
    dev: &B::Device,
) -> Tensor<B, 2> {
    let v = load_f32_vec(st, key);
    assert_eq!(
        v.len(),
        r * c,
        "{key}: expected {r}×{c}={}, got {}",
        r * c,
        v.len()
    );
    Tensor::<B, 1>::from_floats(v.as_slice(), dev).reshape([r, c])
}

fn load_t3<B: Backend>(
    st: &SafeTensors,
    key: &str,
    a: usize,
    b: usize,
    c: usize,
    dev: &B::Device,
) -> Tensor<B, 3> {
    let v = load_f32_vec(st, key);
    assert_eq!(
        v.len(),
        a * b * c,
        "{key}: expected {a}×{b}×{c}={}, got {}",
        a * b * c,
        v.len()
    );
    Tensor::<B, 1>::from_floats(v.as_slice(), dev).reshape([a, b, c])
}

// ═══ Batched LayerNorm ═════════════════════════════════════════════

/// LayerNorm over last dim of [N, S, d]
fn layernorm_3d<B: Backend>(w: &Tensor<B, 2>, b: &Tensor<B, 2>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let [n, s, d] = x.dims();
    let mean = x.clone().mean_dim(2); // [N, S, 1]
    let var = (x.clone() - mean.clone()).powf_scalar(2.0).mean_dim(2);
    let normed = (x - mean) / (var + 1e-5).sqrt();
    // w: [N, d] → [N, 1, d] for broadcast
    let w_exp = w.clone().unsqueeze_dim::<3>(1); // [N, 1, d]
    let b_exp = b.clone().unsqueeze_dim::<3>(1);
    normed * w_exp + b_exp
}

// ═══ Batched Transformer Layer ═════════════════════════════════════

struct BatchedLayer<B: Backend> {
    // Full QKV projection: [300, 3*d_model, d_model]
    attn_in_w: Tensor<B, 3>,
    attn_in_b: Tensor<B, 2>,  // [300, 3*d_model]
    attn_out_w: Tensor<B, 3>, // [300, d_model, d_model]
    attn_out_b: Tensor<B, 2>,
    norm1_w: Tensor<B, 2>, // [300, d_model]
    norm1_b: Tensor<B, 2>,
    ffn_w1: Tensor<B, 3>, // [300, ffn_dim, d_model]
    ffn_b1: Tensor<B, 2>,
    ffn_w2: Tensor<B, 3>, // [300, d_model, ffn_dim]
    ffn_b2: Tensor<B, 2>,
    norm2_w: Tensor<B, 2>,
    norm2_b: Tensor<B, 2>,
}

impl<B: Backend> BatchedLayer<B> {
    fn load(st: &SafeTensors, l: usize, dev: &B::Device) -> Self {
        // v3 keys: attn_in_proj_w.{l}, attn_in_proj_b.{l}, etc. — already [300, ...]
        Self {
            attn_in_w: load_t3::<B>(
                st,
                &format!("attn_in_proj_w.{l}"),
                NUM_BRAINS,
                3 * D_MODEL,
                D_MODEL,
                dev,
            ),
            attn_in_b: load_t2::<B>(
                st,
                &format!("attn_in_proj_b.{l}"),
                NUM_BRAINS,
                3 * D_MODEL,
                dev,
            ),
            attn_out_w: load_t3::<B>(
                st,
                &format!("attn_out_proj_w.{l}"),
                NUM_BRAINS,
                D_MODEL,
                D_MODEL,
                dev,
            ),
            attn_out_b: load_t2::<B>(
                st,
                &format!("attn_out_proj_b.{l}"),
                NUM_BRAINS,
                D_MODEL,
                dev,
            ),
            norm1_w: load_t2::<B>(st, &format!("ln1_w.{l}"), NUM_BRAINS, D_MODEL, dev),
            norm1_b: load_t2::<B>(st, &format!("ln1_b.{l}"), NUM_BRAINS, D_MODEL, dev),
            ffn_w1: load_t3::<B>(
                st,
                &format!("ffn_w1.{l}"),
                NUM_BRAINS,
                FFN_DIM,
                D_MODEL,
                dev,
            ),
            ffn_b1: load_t2::<B>(st, &format!("ffn_b1.{l}"), NUM_BRAINS, FFN_DIM, dev),
            ffn_w2: load_t3::<B>(
                st,
                &format!("ffn_w2.{l}"),
                NUM_BRAINS,
                D_MODEL,
                FFN_DIM,
                dev,
            ),
            ffn_b2: load_t2::<B>(st, &format!("ffn_b2.{l}"), NUM_BRAINS, D_MODEL, dev),
            norm2_w: load_t2::<B>(st, &format!("ln2_w.{l}"), NUM_BRAINS, D_MODEL, dev),
            norm2_b: load_t2::<B>(st, &format!("ln2_b.{l}"), NUM_BRAINS, D_MODEL, dev),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [N, S, d] where N=300, S=seq_len
        let [n, s, d] = x.dims();

        // === Multi-head self-attention ===
        // QKV projection: [N, S, d] @ [N, d, 3d]^T → [N, S, 3d]
        // Implemented via bmm: [N, S, d] @ [N, 3d, d]^T
        let attn_in_w_t = self.attn_in_w.clone().swap_dims(1, 2); // [N, d, 3d]
        let qkv = x.clone().matmul(attn_in_w_t) + self.attn_in_b.clone().unsqueeze_dim::<3>(1); // [N, S, 3d]

        // Split Q, K, V
        let q = qkv.clone().slice([0..n, 0..s, 0..d]);
        let k = qkv.clone().slice([0..n, 0..s, d..2 * d]);
        let v = qkv.slice([0..n, 0..s, 2 * d..3 * d]);

        // Reshape for multi-head: [N, S, d] → [N*H, S, hd]
        let reshape_heads = |t: Tensor<B, 3>| -> Tensor<B, 3> {
            // [N, S, d] → [N, S, H, hd] → [N, H, S, hd] → [N*H, S, hd]
            t.reshape([n, s, NUM_HEADS, HEAD_DIM])
                .swap_dims(1, 2) // [N, H, S, hd]
                .reshape([n * NUM_HEADS, s, HEAD_DIM])
        };

        let q = reshape_heads(q);
        let k = reshape_heads(k);
        let v = reshape_heads(v);

        // Attention: softmax(Q @ K^T / sqrt(hd)) @ V
        let scale = (HEAD_DIM as f64).powf(-0.5);
        let attn_scores = q.matmul(k.transpose()).mul_scalar(scale); // [N*H, S, S]
        let attn_weights = softmax(attn_scores, 2);
        let attn_out = attn_weights.matmul(v); // [N*H, S, hd]

        // Reshape back: [N*H, S, hd] → [N, H, S, hd] → [N, S, H, hd] → [N, S, d]
        let attn_out = attn_out
            .reshape([n, NUM_HEADS, s, HEAD_DIM])
            .swap_dims(1, 2)
            .reshape([n, s, d]);

        // Output projection: [N, S, d] @ [N, d, d]^T
        let out_w_t = self.attn_out_w.clone().swap_dims(1, 2);
        let attn_out = attn_out.matmul(out_w_t) + self.attn_out_b.clone().unsqueeze_dim::<3>(1);

        // Residual + LayerNorm1
        let x = layernorm_3d(&self.norm1_w, &self.norm1_b, x + attn_out);

        // === FFN: gelu(x @ W1^T + b1) @ W2^T + b2 ===
        let w1_t = self.ffn_w1.clone().swap_dims(1, 2); // [N, d, ffn]
        let ff = burn::tensor::activation::gelu(
            x.clone().matmul(w1_t) + self.ffn_b1.clone().unsqueeze_dim::<3>(1),
        );
        let w2_t = self.ffn_w2.clone().swap_dims(1, 2); // [N, ffn, d]
        let ff = ff.matmul(w2_t) + self.ffn_b2.clone().unsqueeze_dim::<3>(1);

        // Residual + LayerNorm2
        layernorm_3d(&self.norm2_w, &self.norm2_b, x + ff)
    }
}

// ═══ Full Batched Temporal Model ═══════════════════════════════════

struct BatchedSwarm<B: Backend> {
    input_proj_w: Tensor<B, 3>, // [300, d, 54]
    input_proj_b: Tensor<B, 2>, // [300, d]
    final_ln_w: Tensor<B, 2>,   // [300, d]
    final_ln_b: Tensor<B, 2>,
    layers: Vec<BatchedLayer<B>>,
    // Gate
    gate_w1: Tensor<B, 2>, // [d, 54]
    gate_b1: Tensor<B, 1>,
    gate_w2: Tensor<B, 2>, // [300, d]
    gate_b2: Tensor<B, 1>,
    // Action head
    action_w: Tensor<B, 2>, // [4, d]
    action_b: Tensor<B, 1>,
    // Confidence head
    conf_w: Tensor<B, 2>, // [1, d]
    conf_b: Tensor<B, 1>,
}

impl<B: Backend> BatchedSwarm<B> {
    fn load(st: &SafeTensors, dev: &B::Device) -> Self {
        eprintln!("[BURN] Loading v3 temporal weights (pre-stacked [300, ...])...");
        let t0 = Instant::now();

        let mut layers = Vec::new();
        for l in 0..NUM_LAYERS {
            eprintln!("[BURN]   Layer {l}...");
            layers.push(BatchedLayer::load(st, l, dev));
        }

        let s = Self {
            input_proj_w: load_t3::<B>(st, "input_proj_w", NUM_BRAINS, D_MODEL, INPUT_DIM, dev),
            input_proj_b: load_t2::<B>(st, "input_proj_b", NUM_BRAINS, D_MODEL, dev),
            final_ln_w: load_t2::<B>(st, "final_ln_w", NUM_BRAINS, D_MODEL, dev),
            final_ln_b: load_t2::<B>(st, "final_ln_b", NUM_BRAINS, D_MODEL, dev),
            layers,
            gate_w1: load_t2::<B>(st, "gate.0.weight", D_MODEL, INPUT_DIM, dev),
            gate_b1: load_t1::<B>(st, "gate.0.bias", D_MODEL, dev),
            gate_w2: load_t2::<B>(st, "gate.2.weight", NUM_BRAINS, D_MODEL, dev),
            gate_b2: load_t1::<B>(st, "gate.2.bias", NUM_BRAINS, dev),
            action_w: load_t2::<B>(st, "action_head.weight", NUM_CLASSES, D_MODEL, dev),
            action_b: load_t1::<B>(st, "action_head.bias", NUM_CLASSES, dev),
            conf_w: load_t2::<B>(st, "confidence_head.weight", 1, D_MODEL, dev),
            conf_b: load_t1::<B>(st, "confidence_head.bias", 1, dev),
        };
        eprintln!("[BURN] ✅ Loaded in {:.1}s", t0.elapsed().as_secs_f64());
        s
    }

    fn forward(&self, seq: Tensor<B, 2>) -> (Vec<f32>, f32) {
        // seq: [S=20, 54] — temporal sequence of features
        let s = SEQ_LEN;
        let d = D_MODEL;
        let n = NUM_BRAINS;

        // Expand sequence to all brains: [S, 54] → [N, S, 54]
        let x = seq.clone().unsqueeze_dim::<3>(0).repeat_dim(0, n); // [300, 20, 54]

        // Input projection: [N, S, 54] @ [N, 54, d]^T → [N, S, d]
        let ip_t = self.input_proj_w.clone().swap_dims(1, 2); // [N, 54, d]
        let mut h = x.matmul(ip_t) + self.input_proj_b.clone().unsqueeze_dim::<3>(1); // [N, S, d]

        // Transformer layers
        for layer in &self.layers {
            h = layer.forward(h);
        }

        // Mean pool over sequence: [N, S, d] → [N, d]
        let h = h.mean_dim(1).squeeze::<2>(1); // [N, d]

        // Final LayerNorm
        let mean = h.clone().mean_dim(1); // [N, 1]
        let var = (h.clone() - mean.clone()).powf_scalar(2.0).mean_dim(1);
        let h = (h.clone() - mean) / (var + 1e-5).sqrt();
        let h = h * self.final_ln_w.clone() + self.final_ln_b.clone(); // [N, d]

        // Gate: softmax(relu(x_last @ W1 + b1) @ W2 + b2) → [1, N]
        // Use last timestep features for gating
        let x_last = seq.slice([s - 1..s, 0..INPUT_DIM]); // [1, 54]
        let gate_h = burn::tensor::activation::relu(
            x_last.matmul(self.gate_w1.clone().transpose()) + self.gate_b1.clone().unsqueeze::<2>(),
        ); // [1, d]
        let gate_logits =
            gate_h.matmul(self.gate_w2.clone().transpose()) + self.gate_b2.clone().unsqueeze::<2>();
        let gate_weights = softmax(gate_logits, 1); // [1, N]

        // Weighted combination: [1, N] @ [N, d] → [1, d]
        let combined = gate_weights.matmul(h); // [1, d]

        // Action head
        let action_logits = combined.clone().matmul(self.action_w.clone().transpose())
            + self.action_b.clone().unsqueeze::<2>();
        let action_probs = softmax(action_logits, 1);

        // Confidence head
        let conf = burn::tensor::activation::sigmoid(
            combined.matmul(self.conf_w.clone().transpose()) + self.conf_b.clone().unsqueeze::<2>(),
        );

        let a: Vec<f32> = action_probs.to_data().to_vec().unwrap();
        let c: f32 = conf.to_data().to_vec::<f32>().unwrap()[0];
        (a, c)
    }
}

// ═══ Feature Ring Buffer ═══════════════════════════════════════════

struct FeatureRingBuffer {
    buffer: HashMap<String, Vec<[f32; INPUT_DIM]>>, // symbol → last N feature vectors
}

impl FeatureRingBuffer {
    fn new() -> Self {
        let mut buffer = HashMap::new();
        for sym in &SYMBOL_MAP {
            buffer.insert(sym.to_string(), Vec::with_capacity(SEQ_LEN + 10));
        }
        Self { buffer }
    }

    fn push(&mut self, sym: &str, features: [f32; INPUT_DIM]) {
        if let Some(buf) = self.buffer.get_mut(sym) {
            buf.push(features);
            // Keep last SEQ_LEN * 2 to avoid frequent realloc
            if buf.len() > SEQ_LEN * 2 {
                let start = buf.len() - SEQ_LEN;
                *buf = buf[start..].to_vec();
            }
        }
    }

    fn get_sequence(&self, sym: &str) -> Option<Vec<[f32; INPUT_DIM]>> {
        let buf = self.buffer.get(sym)?;
        if buf.len() < SEQ_LEN {
            return None; // Not enough history yet
        }
        let start = buf.len() - SEQ_LEN;
        Some(buf[start..].to_vec())
    }
}

// ═══ Output structures ═════════════════════════════════════════════

#[derive(Serialize)]
struct Prediction {
    symbol: String,
    action: String,
    action_probs: HashMap<String, f32>,
    confidence: f32,
    inference_ms: f64,
    seq_len: usize,
    top_active_brains: usize,
}

#[derive(Serialize)]
struct Output {
    tick: u64,
    timestamp: String,
    predictions: HashMap<String, Prediction>,
    model: String,
    total_inference_ms: f64,
}

// ═══ Feature Collection ════════════════════════════════════════════

fn load_norm_stats(model_dir: &str) -> ([f32; INPUT_DIM], [f32; INPUT_DIM]) {
    fn load_npy(path: &str) -> [f32; INPUT_DIM] {
        let data = fs::read(path).unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));
        let header_len = u16::from_le_bytes([data[8], data[9]]) as usize;
        let start = 10 + header_len;
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                data[start..].as_ptr() as *const f32,
                (data.len() - start) / 4,
            )
        };
        let mut arr = [0.0f32; INPUT_DIM];
        for i in 0..INPUT_DIM.min(floats.len()) {
            arr[i] = floats[i];
        }
        arr
    }
    (
        load_npy(&format!("{model_dir}/norm_means.npy")),
        load_npy(&format!("{model_dir}/norm_stds.npy")),
    )
}

fn read_features_snapshot(workspace: &str) -> Option<HashMap<String, [f32; INPUT_DIM]>> {
    let path = format!("{workspace}/quant/burn_features_cache.json");
    let data = fs::read_to_string(&path).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&data).ok()?;
    let mut result = HashMap::new();
    for sym in &SYMBOL_MAP {
        let arr = parsed.get(*sym)?.as_array()?;
        let mut features = [0.0f32; INPUT_DIM];
        for (i, v) in arr.iter().enumerate().take(INPUT_DIM) {
            features[i] = v.as_f64()? as f32;
        }
        result.insert(sym.to_string(), features);
    }
    Some(result)
}

fn normalize_features(
    raw: &[f32; INPUT_DIM],
    means: &[f32; INPUT_DIM],
    stds: &[f32; INPUT_DIM],
) -> [f32; INPUT_DIM] {
    let mut out = [0.0f32; INPUT_DIM];
    for i in 0..INPUT_DIM {
        let s = if stds[i].abs() < 1e-8 { 1.0 } else { stds[i] };
        out[i] = (raw[i] - means[i]) / s;
    }
    out
}

// ═══ Main ══════════════════════════════════════════════════════════

type MyBackend = Wgpu;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).map(|s| s.as_str()).unwrap_or("./weights");
    let workspace = args.get(2).map(|s| s.as_str()).unwrap_or(".");
    let interval_s: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    let single_shot = args.iter().any(|a| a == "--once");

    eprintln!("[BURN] Joi Burn v3 — TEMPORAL 300-Brain Swarm");
    eprintln!("[BURN] {NUM_BRAINS} brains × {NUM_LAYERS}L × {NUM_HEADS}H × d={D_MODEL}");
    eprintln!("[BURN] Sequence length: {SEQ_LEN}, Input dim: {INPUT_DIM}");

    let device = WgpuDevice::default();
    eprintln!("[BURN] Device: WGPU (Metal)");

    let st_path = format!("{model_dir}/best_model_fp32.safetensors");
    eprintln!("[BURN] Loading: {st_path}");
    let t0 = Instant::now();
    let st_data = fs::read(&st_path).expect("Cannot read safetensors");
    let st = SafeTensors::deserialize(&st_data).expect("Cannot parse safetensors");

    let model = BatchedSwarm::<MyBackend>::load(&st, &device);

    let (norm_means, norm_stds) = load_norm_stats(model_dir);
    eprintln!("[BURN] Normalization loaded");

    // Warmup with dummy sequence
    eprintln!("[BURN] Warmup...");
    let dummy = Tensor::<MyBackend, 2>::zeros([SEQ_LEN, INPUT_DIM], &device);
    let t0 = Instant::now();
    let _ = model.forward(dummy);
    let warmup_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[BURN] Warmup: {warmup_ms:.1}ms");

    // Benchmark
    let t0 = Instant::now();
    for _ in 0..6 {
        let x = Tensor::<MyBackend, 2>::zeros([SEQ_LEN, INPUT_DIM], &device);
        let _ = model.forward(x);
    }
    let bench_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[BURN] 🔥 6-symbol benchmark: {bench_ms:.1}ms ({:.1}ms/sym)",
        bench_ms / 6.0
    );

    if single_shot {
        return;
    }

    // Ring buffer for temporal sequences
    let mut ring = FeatureRingBuffer::new();
    let mut tick: u64 = 0;
    let output_path = format!("{workspace}/burn_brain_signal.json");

    eprintln!("[BURN] Collecting {SEQ_LEN} snapshots before first prediction...");

    loop {
        tick += 1;

        // Read latest feature snapshot
        if let Some(snapshot) = read_features_snapshot(workspace) {
            for (sym, raw) in &snapshot {
                let normed = normalize_features(raw, &norm_means, &norm_stds);
                ring.push(sym, normed);
            }
        } else {
            if tick % 10 == 0 {
                eprintln!("[BURN] tick {tick}: no features, waiting...");
            }
            std::thread::sleep(std::time::Duration::from_secs(interval_s));
            continue;
        }

        // Check if we have enough history
        let ready = SYMBOL_MAP.iter().all(|s| ring.get_sequence(s).is_some());
        if !ready {
            if tick % 5 == 0 {
                let counts: Vec<String> = SYMBOL_MAP
                    .iter()
                    .map(|s| format!("{}={}", s, ring.buffer.get(*s).map_or(0, |b| b.len())))
                    .collect();
                eprintln!("[BURN] tick {tick}: buffering... {}", counts.join(" "));
            }
            std::thread::sleep(std::time::Duration::from_secs(interval_s));
            continue;
        }

        let t_start = Instant::now();
        let mut predictions = HashMap::new();

        for sym in &SYMBOL_MAP {
            let seq = ring.get_sequence(sym).unwrap();
            // Flatten [20][54] → [20, 54] tensor
            let mut flat = Vec::with_capacity(SEQ_LEN * INPUT_DIM);
            for step in &seq {
                flat.extend_from_slice(step);
            }
            let tensor = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), &device)
                .reshape([SEQ_LEN, INPUT_DIM]);

            let t_sym = Instant::now();
            let (action_data, conf) = model.forward(tensor);
            let sym_ms = t_sym.elapsed().as_secs_f64() * 1000.0;

            let action_idx = action_data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            let mut action_probs = HashMap::new();
            for (j, name) in CLASS_NAMES.iter().enumerate() {
                action_probs.insert(name.to_string(), action_data[j]);
            }

            predictions.insert(
                sym.to_string(),
                Prediction {
                    symbol: sym.to_string(),
                    action: CLASS_NAMES[action_idx].to_string(),
                    action_probs,
                    confidence: conf,
                    inference_ms: sym_ms,
                    seq_len: SEQ_LEN,
                    top_active_brains: NUM_BRAINS,
                },
            );
        }

        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let output = Output {
            tick,
            timestamp: chrono::Utc::now().to_rfc3339(),
            predictions,
            model: "burn_swarm_v3_300b_temporal_metal".to_string(),
            total_inference_ms: total_ms,
        };

        if let Ok(json) = serde_json::to_string_pretty(&output) {
            let _ = fs::write(&output_path, json);
        }

        if tick % 60 == 0 || tick <= 3 {
            let syms: Vec<String> = SYMBOL_MAP
                .iter()
                .map(|s| {
                    let p = output.predictions.get(*s).unwrap();
                    format!(
                        "{s}={} {:.0}%",
                        &p.action[..3.min(p.action.len())],
                        p.confidence * 100.0
                    )
                })
                .collect();
            eprintln!("[BURN] tick {tick}: {total_ms:.0}ms | {}", syms.join(" "));
        }

        std::thread::sleep(std::time::Duration::from_secs(interval_s));
    }
}
