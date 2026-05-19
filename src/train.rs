//! Joi Burn Training — 300-brain batched swarm training via Burn autodiff
//!
//! Uses WGPU backend (Metal on Mac, Vulkan on Linux) — bypasses CUDA driver bugs.
//! All 300 brains train in parallel via batched matmul.
//!
//! Usage: joi-train --db /path/to/joi_brain.db --epochs 40 --batch-size 16 --output ./models/

use burn::backend::wgpu::{init_setup, MemoryConfiguration, RuntimeOptions, Vulkan, WgpuDevice};
use burn::backend::{Autodiff, Wgpu};
use burn::module::Param;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{BinFileRecorder, FullPrecisionSettings};
use burn::tensor::activation::{gelu, softmax};
use clap::Parser;
use rusqlite::Connection;
use std::time::Instant;

const INPUT_DIM: usize = 54;
const SEQ_LEN: usize = 20;
const NUM_CLASSES: usize = 4;

// ═══ CLI Args ══════════════════════════════════════════════════════

#[derive(Parser)]
struct Args {
    #[arg(long)]
    db: String,
    #[arg(long, default_value = "./burn_models_v3")]
    output: String,
    #[arg(long, default_value_t = 40)]
    epochs: usize,
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    #[arg(long, default_value_t = 300)]
    n_brains: usize,
    #[arg(long, default_value_t = 256)]
    d_model: usize,
    #[arg(long, default_value_t = 8)]
    n_heads: usize,
    #[arg(long, default_value_t = 4)]
    n_layers: usize,
    #[arg(long, default_value_t = 1024)]
    ffn_dim: usize,
    #[arg(long, default_value_t = 1e-4)]
    lr: f64,
}

// ═══ Data Loading ══════════════════════════════════════════════════

struct Sample {
    features: Vec<f32>, // [seq_len * input_dim] flattened
    label: usize,
}

fn load_data(db_path: &str) -> (Vec<Sample>, Vec<Sample>, [f32; INPUT_DIM], [f32; INPUT_DIM]) {
    let conn = Connection::open(db_path).expect("Cannot open DB");
    let symbols = ["BTC", "ETH", "SOL", "HYPE", "BNB", "LINK"];
    let sym_onehot: std::collections::HashMap<&str, usize> =
        symbols.iter().enumerate().map(|(i, s)| (*s, i)).collect();

    let vote_cols = [
        "ob_vote",
        "tape_vote",
        "mdf_vote",
        "funding_vote",
        "volatility_vote",
        "entropy_vote",
        "markov_vote",
        "correlation_vote",
        "regime_vote",
        "resonance_vote",
        "binance_vote",
        "hyperliquid_vote",
        "oi_delta_vote",
        "whale_vote",
        "cross_ex_vote",
        "mrf_vote",
        "spread_vote",
        "liq_cascade_vote",
        "sentiment_vel_vote",
        "max_pain_vote",
        "fourier_vote",
        "ornstein_uhlenbeck_vote",
        "shannon_vote",
        "bayesian_vote",
        "pca_vote",
        "drift_diffusion_vote",
        "mutual_info_vote",
        "copula_vote",
        "dax_modifier",
    ];
    let extra_cols = [
        "mdt_confirmed",
        "mdt_momentum_bps",
        "omega",
        "effective_omega",
        "agreement",
        "num_voters",
        "mrf_confidence",
        "spread_bps",
        "price",
        "mark_price",
        "best_bid",
    ];

    let mut all_samples: Vec<Sample> = Vec::new();

    for sym in &symbols {
        eprintln!("  Loading {sym}...");
        let col_list: String = std::iter::once("ts".to_string())
            .chain(vote_cols.iter().map(|s| s.to_string()))
            .chain(extra_cols.iter().map(|s| s.to_string()))
            .collect::<Vec<_>>()
            .join(",");

        let query = format!(
            "SELECT {} FROM signal_snapshots WHERE symbol=? ORDER BY ts ASC",
            col_list
        );
        let mut stmt = conn.prepare(&query).expect("Bad query");
        let rows: Vec<(f64, Vec<f64>)> = stmt
            .query_map([sym], |row| {
                let ts: f64 = row.get(0)?;
                let mut vals = Vec::with_capacity(vote_cols.len() + extra_cols.len());
                for i in 1..=(vote_cols.len() + extra_cols.len()) {
                    vals.push(row.get::<_, f64>(i).unwrap_or(0.0));
                }
                Ok((ts, vals))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        if rows.len() < SEQ_LEN {
            eprintln!("    skip ({} rows)", rows.len());
            continue;
        }

        // Build feature vectors with one-hot
        let mut all_ts: Vec<f64> = Vec::with_capacity(rows.len());
        let mut all_feats: Vec<Vec<f32>> = Vec::with_capacity(rows.len());

        for (ts, vals) in &rows {
            all_ts.push(*ts);
            let mut feat = Vec::with_capacity(INPUT_DIM);
            // First 40: signal votes + extras (truncated/padded)
            for i in 0..40 {
                feat.push(if i < vals.len() { vals[i] as f32 } else { 0.0 });
            }
            // 6: one-hot symbol
            for i in 0..6 {
                feat.push(if i == sym_onehot[sym] { 1.0 } else { 0.0 });
            }
            // 4: funding features
            let funding_vote = if vals.len() > 3 { vals[3] as f32 } else { 0.0 };
            let spread_bps = if vals.len() > 36 {
                vals[36] as f32
            } else {
                0.0
            };
            let mrf_conf = if vals.len() > 35 {
                vals[35] as f32
            } else {
                0.0
            };
            let agreement = if vals.len() > 33 {
                vals[33] as f32
            } else {
                0.0
            };
            feat.extend_from_slice(&[funding_vote, spread_bps, mrf_conf, agreement]);
            // 4: orderbook features
            let ob_vote = if !vals.is_empty() {
                vals[0] as f32
            } else {
                0.0
            };
            let best_bid = if vals.len() > 39 {
                vals[39] as f32
            } else {
                0.0
            };
            let price = if vals.len() > 37 {
                vals[37] as f32
            } else {
                0.0
            };
            let mark_price = if vals.len() > 38 {
                vals[38] as f32
            } else {
                0.0
            };
            feat.extend_from_slice(&[ob_vote, best_bid, price, mark_price]);

            all_feats.push(feat);
        }

        // Get labels from omega_history
        let label_query = "SELECT ts, action FROM omega_history WHERE symbol=? ORDER BY ts ASC";
        let mut lstmt = conn.prepare(label_query).expect("Bad label query");
        let labels: Vec<(f64, String)> = lstmt
            .query_map([sym], |row| {
                Ok((row.get::<_, f64>(0)?, row.get::<_, String>(1)?))
            })
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();

        let label_ts: Vec<f64> = labels.iter().map(|l| l.0).collect();
        let label_actions: Vec<&str> = labels.iter().map(|l| l.1.as_str()).collect();

        let mut seq_count = 0;
        for anchor_idx in (SEQ_LEN..all_ts.len()).step_by(5) {
            let anchor_t = all_ts[anchor_idx];
            // Find closest label
            if label_ts.is_empty() {
                continue;
            }
            let li = label_ts.partition_point(|&t| t <= anchor_t);
            let li = if li > 0 { li - 1 } else { continue };
            if (label_ts[li] - anchor_t).abs() > 5.0 {
                continue;
            }
            let label = match label_actions[li] {
                "OBSERVE" => 0,
                "HOLD" => 1,
                "ENTER_LONG" => 2,
                "ENTER_SHORT" => 3,
                _ => continue,
            };

            // Build sequence [seq_len, input_dim]
            if anchor_idx < SEQ_LEN {
                continue;
            }
            let start = anchor_idx + 1 - SEQ_LEN;
            let mut seq_flat = Vec::with_capacity(SEQ_LEN * INPUT_DIM);
            for i in start..=anchor_idx {
                seq_flat.extend_from_slice(&all_feats[i]);
            }

            all_samples.push(Sample {
                features: seq_flat,
                label,
            });
            seq_count += 1;
        }
        eprintln!("    {seq_count} sequences from {} snapshots", rows.len());
    }

    // Shuffle and split
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    all_samples.shuffle(&mut rng);

    let val_n = all_samples.len() / 10;
    let val: Vec<Sample> = all_samples.drain(..val_n).collect();
    let train = all_samples;

    // Compute normalization stats from training data
    let n = train.len() * SEQ_LEN;
    let mut means = [0.0f64; INPUT_DIM];
    let mut vars = [0.0f64; INPUT_DIM];

    for s in &train {
        for t in 0..SEQ_LEN {
            for d in 0..INPUT_DIM {
                means[d] += s.features[t * INPUT_DIM + d] as f64;
            }
        }
    }
    for d in 0..INPUT_DIM {
        means[d] /= n as f64;
    }

    for s in &train {
        for t in 0..SEQ_LEN {
            for d in 0..INPUT_DIM {
                let diff = s.features[t * INPUT_DIM + d] as f64 - means[d];
                vars[d] += diff * diff;
            }
        }
    }
    for d in 0..INPUT_DIM {
        vars[d] = (vars[d] / n as f64).sqrt().max(1e-8);
    }

    let mut nm = [0.0f32; INPUT_DIM];
    let mut ns = [0.0f32; INPUT_DIM];
    for d in 0..INPUT_DIM {
        nm[d] = means[d] as f32;
        ns[d] = vars[d] as f32;
    }

    eprintln!(
        "  Train: {} sequences, Val: {} sequences",
        train.len(),
        val.len()
    );
    (train, val, nm, ns)
}

// ═══ Batched Brain Model (Burn Module) ═════════════════════════════

#[derive(Module, Debug)]
struct BatchedBrainSwarm<B: Backend> {
    // Input projection [n_brains, d_model, input_dim]
    input_proj_w: Param<Tensor<B, 3>>,
    input_proj_b: Param<Tensor<B, 2>>,
    // Per-layer weights (flattened — indexed by layer)
    attn_in_w: Param<Tensor<B, 4>>,  // [n_layers, n_brains, 3*d, d]
    attn_in_b: Param<Tensor<B, 3>>,  // [n_layers, n_brains, 3*d]
    attn_out_w: Param<Tensor<B, 4>>, // [n_layers, n_brains, d, d]
    attn_out_b: Param<Tensor<B, 3>>, // [n_layers, n_brains, d]
    ffn_w1: Param<Tensor<B, 4>>,     // [n_layers, n_brains, ffn, d]
    ffn_b1: Param<Tensor<B, 3>>,     // [n_layers, n_brains, ffn]
    ffn_w2: Param<Tensor<B, 4>>,     // [n_layers, n_brains, d, ffn]
    ffn_b2: Param<Tensor<B, 3>>,     // [n_layers, n_brains, d]
    ln1_w: Param<Tensor<B, 3>>,      // [n_layers, n_brains, d]
    ln1_b: Param<Tensor<B, 3>>,
    ln2_w: Param<Tensor<B, 3>>,
    ln2_b: Param<Tensor<B, 3>>,
    final_ln_w: Param<Tensor<B, 2>>, // [n_brains, d]
    final_ln_b: Param<Tensor<B, 2>>,
    // Gate
    gate_w1: Param<Tensor<B, 2>>, // [d_model, input_dim]
    gate_b1: Param<Tensor<B, 1>>,
    gate_w2: Param<Tensor<B, 2>>, // [n_brains, d_model]
    gate_b2: Param<Tensor<B, 1>>,
    // Heads
    action_w: Param<Tensor<B, 2>>, // [n_classes, d_model]
    action_b: Param<Tensor<B, 1>>,
    conf_w: Param<Tensor<B, 2>>,
    conf_b: Param<Tensor<B, 1>>,
    // Config (not params)
    n_brains: usize,
    n_layers: usize,
    n_heads: usize,
    d_model: usize,
}

impl<B: Backend> BatchedBrainSwarm<B> {
    fn new(
        n_brains: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        ffn_dim: usize,
        n_classes: usize,
        device: &B::Device,
    ) -> Self {
        let si = (INPUT_DIM as f64).powf(-0.5) as f32;
        let sd = (d_model as f64).powf(-0.5) as f32;
        let sf = (ffn_dim as f64).powf(-0.5) as f32;

        let randn3 = |d0, d1, d2, s: f32| -> Tensor<B, 3> {
            Tensor::random(
                [d0, d1, d2],
                burn::tensor::Distribution::Normal(0.0, s as f64),
                device,
            )
        };
        let randn4 = |d0, d1, d2, d3, s: f32| -> Tensor<B, 4> {
            Tensor::random(
                [d0, d1, d2, d3],
                burn::tensor::Distribution::Normal(0.0, s as f64),
                device,
            )
        };
        let zeros2 = |d0, d1| Tensor::<B, 2>::zeros([d0, d1], device);
        let zeros3 = |d0, d1, d2| Tensor::<B, 3>::zeros([d0, d1, d2], device);
        let ones3 = |d0, d1, d2| Tensor::<B, 3>::ones([d0, d1, d2], device);
        let zeros1 = |d| Tensor::<B, 1>::zeros([d], device);

        Self {
            input_proj_w: Param::from_tensor(randn3(n_brains, d_model, INPUT_DIM, si)),
            input_proj_b: Param::from_tensor(zeros2(n_brains, d_model)),
            attn_in_w: Param::from_tensor(randn4(n_layers, n_brains, 3 * d_model, d_model, sd)),
            attn_in_b: Param::from_tensor(zeros3(n_layers, n_brains, 3 * d_model)),
            attn_out_w: Param::from_tensor(randn4(n_layers, n_brains, d_model, d_model, sd)),
            attn_out_b: Param::from_tensor(zeros3(n_layers, n_brains, d_model)),
            ffn_w1: Param::from_tensor(randn4(n_layers, n_brains, ffn_dim, d_model, sd)),
            ffn_b1: Param::from_tensor(zeros3(n_layers, n_brains, ffn_dim)),
            ffn_w2: Param::from_tensor(randn4(n_layers, n_brains, d_model, ffn_dim, sf)),
            ffn_b2: Param::from_tensor(zeros3(n_layers, n_brains, d_model)),
            ln1_w: Param::from_tensor(ones3(n_layers, n_brains, d_model)),
            ln1_b: Param::from_tensor(zeros3(n_layers, n_brains, d_model)),
            ln2_w: Param::from_tensor(ones3(n_layers, n_brains, d_model)),
            ln2_b: Param::from_tensor(zeros3(n_layers, n_brains, d_model)),
            final_ln_w: Param::from_tensor(Tensor::ones([n_brains, d_model], device)),
            final_ln_b: Param::from_tensor(zeros2(n_brains, d_model)),
            gate_w1: Param::from_tensor(Tensor::random(
                [d_model, INPUT_DIM],
                burn::tensor::Distribution::Normal(0.0, si as f64),
                device,
            )),
            gate_b1: Param::from_tensor(zeros1(d_model)),
            gate_w2: Param::from_tensor(Tensor::random(
                [n_brains, d_model],
                burn::tensor::Distribution::Normal(0.0, sd as f64),
                device,
            )),
            gate_b2: Param::from_tensor(zeros1(n_brains)),
            action_w: Param::from_tensor(Tensor::random(
                [n_classes, d_model],
                burn::tensor::Distribution::Normal(0.0, sd as f64),
                device,
            )),
            action_b: Param::from_tensor(zeros1(n_classes)),
            conf_w: Param::from_tensor(Tensor::random(
                [1, d_model],
                burn::tensor::Distribution::Normal(0.0, sd as f64),
                device,
            )),
            conf_b: Param::from_tensor(zeros1(1)),
            n_brains,
            n_layers,
            n_heads,
            d_model,
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        // x: [B, seq_len, input_dim]
        let [batch, seq, _] = x.dims();
        let n = self.n_brains;
        let d = self.d_model;
        let hd = d / self.n_heads;

        // Expand for all brains: [N, B, S, input_dim]
        let x_exp = x.clone().unsqueeze_dim::<4>(0).repeat_dim(0, n); // [N, B, S, 54]

        // Input projection: bmm [N, B*S, 54] @ [N, 54, d] -> [N, B*S, d]
        let x_flat = x_exp.reshape([n, batch * seq, INPUT_DIM]);
        let mut h = x_flat.matmul(self.input_proj_w.val().transpose())
            + self.input_proj_b.val().unsqueeze_dim::<3>(1); // [N, B*S, d]
        let h = h.reshape([n, batch, seq, d]); // [N, B, S, d]

        let mut h = h;
        for l in 0..self.n_layers {
            // Extract per-layer weights: slice along dim 0
            let aiw = self.attn_in_w.val().slice([l..l + 1]).squeeze::<3>(0); // [N, 3d, d]
            let aib: Tensor<B, 2> = self.attn_in_b.val().slice([l..l + 1]).squeeze(0);
            let aow = self.attn_out_w.val().slice([l..l + 1]).squeeze::<3>(0);
            let aob: Tensor<B, 2> = self.attn_out_b.val().slice([l..l + 1]).squeeze(0);
            let fw1 = self.ffn_w1.val().slice([l..l + 1]).squeeze::<3>(0);
            let fb1: Tensor<B, 2> = self.ffn_b1.val().slice([l..l + 1]).squeeze(0);
            let fw2 = self.ffn_w2.val().slice([l..l + 1]).squeeze::<3>(0);
            let fb2: Tensor<B, 2> = self.ffn_b2.val().slice([l..l + 1]).squeeze(0);
            let l1w: Tensor<B, 2> = self.ln1_w.val().slice([l..l + 1]).squeeze(0);
            let l1b: Tensor<B, 2> = self.ln1_b.val().slice([l..l + 1]).squeeze(0);
            let l2w: Tensor<B, 2> = self.ln2_w.val().slice([l..l + 1]).squeeze(0);
            let l2b: Tensor<B, 2> = self.ln2_b.val().slice([l..l + 1]).squeeze(0);

            // Self-attention
            let h_flat = h.clone().reshape([n, batch * seq, d]);
            let qkv = h_flat.clone().matmul(aiw.transpose()) + aib.unsqueeze_dim::<3>(1);
            // qkv: [N, B*S, 3d] -> split Q, K, V
            let qkv_4d = qkv.reshape([n, batch, seq, 3 * d]);
            let q = qkv_4d.clone().slice([0..n, 0..batch, 0..seq, 0..d]);
            let k = qkv_4d.clone().slice([0..n, 0..batch, 0..seq, d..2 * d]);
            let v = qkv_4d.slice([0..n, 0..batch, 0..seq, 2 * d..3 * d]);

            // Multi-head: reshape to [N*B*H, S, hd]
            let nh = self.n_heads;
            let q = q
                .reshape([n * batch, seq, nh, hd])
                .swap_dims(1, 2)
                .reshape([n * batch * nh, seq, hd]);
            let k = k
                .reshape([n * batch, seq, nh, hd])
                .swap_dims(1, 2)
                .reshape([n * batch * nh, seq, hd]);
            let v = v
                .reshape([n * batch, seq, nh, hd])
                .swap_dims(1, 2)
                .reshape([n * batch * nh, seq, hd]);

            let scale = (hd as f64).powf(-0.5);
            let attn = q.matmul(k.transpose()).mul_scalar(scale);
            let attn = softmax(attn, 2);
            let attn_out = attn.matmul(v); // [N*B*H, S, hd]

            // Reshape back: [N, B, S, d]
            let attn_out = attn_out
                .reshape([n * batch, nh, seq, hd])
                .swap_dims(1, 2)
                .reshape([n, batch * seq, d]);
            let attn_out = attn_out.matmul(aow.transpose()) + aob.unsqueeze_dim::<3>(1);
            let attn_out = attn_out.reshape([n, batch, seq, d]);

            // Residual + LayerNorm1
            let h_res = h + attn_out;
            let mean = h_res.clone().mean_dim(3);
            let var = (h_res.clone() - mean.clone()).powf_scalar(2.0).mean_dim(3);
            let h_norm = (h_res.clone() - mean) / (var + 1e-5).sqrt();
            h = h_norm * l1w.reshape([n, 1, 1, d]) + l1b.reshape([n, 1, 1, d]);

            // FFN: gelu(h @ W1^T + b1) @ W2^T + b2
            let h_flat = h.clone().reshape([n, batch * seq, d]);
            let ff = gelu(h_flat.clone().matmul(fw1.transpose()) + fb1.unsqueeze_dim::<3>(1));
            let ff = ff.matmul(fw2.transpose()) + fb2.unsqueeze_dim::<3>(1);
            let ff = ff.reshape([n, batch, seq, d]);

            // Residual + LayerNorm2
            let h_res = h + ff;
            let mean = h_res.clone().mean_dim(3);
            let var = (h_res.clone() - mean.clone()).powf_scalar(2.0).mean_dim(3);
            let h_norm = (h_res.clone() - mean) / (var + 1e-5).sqrt();
            h = h_norm * l2w.reshape([n, 1, 1, d]) + l2b.reshape([n, 1, 1, d]);
        }

        // Mean pool over sequence: [N, B, S, d] -> [N, B, d]
        let h = h.mean_dim(2).squeeze::<3>(2);

        // Final LayerNorm: h is [N, B, d]
        let mean = h.clone().mean_dim(2); // [N, B, 1]
        let var = (h.clone() - mean.clone()).powf_scalar(2.0).mean_dim(2);
        let h_norm = (h - mean) / (var + 1e-5).sqrt(); // [N, B, d]
        let ln_w = self.final_ln_w.val().unsqueeze_dim::<3>(1); // [N, 1, d]
        let ln_b = self.final_ln_b.val().unsqueeze_dim::<3>(1);
        let h = h_norm.mul(ln_w).add(ln_b); // [N, B, d]

        // brain_outputs: [B, N, d]
        let brain_out = h.swap_dims(0, 1);

        // Gate: latest features
        let x_latest = x
            .slice([0..batch, seq - 1..seq, 0..INPUT_DIM])
            .squeeze::<2>(1); // [B, 54]
        let gate_h = burn::tensor::activation::relu(
            x_latest
                .clone()
                .matmul(self.gate_w1.val().transpose())
                .add(self.gate_b1.val().unsqueeze_dim::<2>(0)),
        );
        let gate_logits = gate_h
            .matmul(self.gate_w2.val().transpose())
            .add(self.gate_b2.val().unsqueeze_dim::<2>(0));
        let gate_w = softmax(gate_logits, 1); // [B, N]

        // Weighted combination: [B, 1, N] @ [B, N, d] -> [B, 1, d] -> [B, d]
        let gate_3d = gate_w.unsqueeze_dim::<3>(1); // [B, 1, N]
        let combined = gate_3d.matmul(brain_out).squeeze::<2>(1);

        // Action logits
        combined.matmul(self.action_w.val().transpose()) + self.action_b.val().unsqueeze_dim::<2>(0)
    }
}

// ═══ Training Loop ═════════════════════════════════════════════════

type TrainBackend = Autodiff<Wgpu>;

fn main() {
    let args = Args::parse();
    // Use ExclusivePages memory strategy for large models (fixes 5090 pool allocation failure)
    let device = WgpuDevice::default();
    let options = RuntimeOptions {
        memory_config: MemoryConfiguration::ExclusivePages,
        ..Default::default()
    };
    let _setup = init_setup::<Vulkan>(&device, options);

    eprintln!(
        "[TRAIN] Joi Burn Training — Batched {}-Brain Swarm",
        args.n_brains
    );
    eprintln!("[TRAIN] Device: WGPU (Vulkan, ExclusivePages memory)");

    // Load data
    let (train_data, val_data, norm_means, norm_stds) = load_data(&args.db);
    eprintln!(
        "[TRAIN] Train: {}, Val: {}",
        train_data.len(),
        val_data.len()
    );

    // Build model
    eprintln!("[TRAIN] Building model on GPU...");
    let model = BatchedBrainSwarm::<TrainBackend>::new(
        args.n_brains,
        args.d_model,
        args.n_heads,
        args.n_layers,
        args.ffn_dim,
        NUM_CLASSES,
        &device,
    );
    let ip_shape = model.input_proj_w.val().shape();
    let n_params_approx = args.n_brains * (args.d_model * INPUT_DIM + args.d_model  // input_proj
        + args.n_layers * (3 * args.d_model * args.d_model + args.d_model * args.d_model  // attn
            + args.ffn_dim * args.d_model + args.d_model * args.ffn_dim  // ffn
            + 4 * args.d_model)  // layernorms
        + args.d_model)  // final ln
        + args.d_model * INPUT_DIM + args.d_model + args.n_brains * args.d_model + args.n_brains  // gate
        + NUM_CLASSES * args.d_model + NUM_CLASSES + args.d_model + 1; // heads
    eprintln!(
        "[TRAIN] Model built — ~{:.0}M params",
        n_params_approx as f64 / 1e6
    );

    // Optimizer
    let optim_config = AdamWConfig::new().with_weight_decay(0.01);
    let mut optim = optim_config.init();

    let out_dir = std::path::Path::new(&args.output);
    std::fs::create_dir_all(out_dir).ok();

    let mut best_val_loss = f64::MAX;
    let mut best_epoch = 0;
    let start = Instant::now();

    for epoch in 1..=args.epochs {
        // Training
        let mut model = model.clone(); // Clone for mutability in training
        let mut train_loss_sum = 0.0f64;
        let mut train_correct = 0usize;
        let mut train_total = 0usize;
        let n_batches = (train_data.len() + args.batch_size - 1) / args.batch_size;

        for batch_idx in 0..n_batches {
            let start_i = batch_idx * args.batch_size;
            let end_i = (start_i + args.batch_size).min(train_data.len());
            let bs = end_i - start_i;

            // Build batch tensors
            let mut feat_data = Vec::with_capacity(bs * SEQ_LEN * INPUT_DIM);
            let mut label_data = Vec::with_capacity(bs);

            for i in start_i..end_i {
                // Normalize features
                for t in 0..SEQ_LEN {
                    for d in 0..INPUT_DIM {
                        let raw = train_data[i].features[t * INPUT_DIM + d];
                        feat_data.push((raw - norm_means[d]) / norm_stds[d]);
                    }
                }
                label_data.push(train_data[i].label as i64);
            }

            let x = Tensor::<TrainBackend, 1>::from_floats(feat_data.as_slice(), &device)
                .reshape([bs, SEQ_LEN, INPUT_DIM]);
            let targets = Tensor::<TrainBackend, 1, Int>::from_ints(
                label_data
                    .iter()
                    .map(|&l| l as i32)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &device,
            );

            // Forward
            let logits = model.forward(x);

            // Cross-entropy loss
            let loss =
                CrossEntropyLoss::new(None, &device).forward(logits.clone(), targets.clone());

            // Backward
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(args.lr, model, grads_params);

            // Track metrics
            let loss_val: f32 = loss.clone().into_data().to_vec().unwrap()[0];
            train_loss_sum += loss_val as f64 * bs as f64;

            let preds = logits.argmax(1);
            let correct: i64 = preds
                .equal(targets.unsqueeze_dim(1))
                .int()
                .sum()
                .into_scalar()
                .elem();
            train_correct += correct as usize;
            train_total += bs;

            if batch_idx == 0 || (batch_idx + 1) % 50 == 0 {
                let elapsed = start.elapsed().as_secs();
                eprintln!(
                    "  E{epoch} {}/{n_batches} loss={loss_val:.4} acc={:.3} [{elapsed}s]",
                    batch_idx + 1,
                    train_correct as f64 / train_total as f64
                );
            }
        }

        // Validation (use inner backend for inference)
        let mut val_loss_sum = 0.0f64;
        let mut val_correct = 0usize;
        let mut val_total = 0usize;
        let val_batches = (val_data.len() + args.batch_size - 1) / args.batch_size;

        for batch_idx in 0..val_batches {
            let start_i = batch_idx * args.batch_size;
            let end_i = (start_i + args.batch_size).min(val_data.len());
            let bs = end_i - start_i;

            let mut feat_data = Vec::with_capacity(bs * SEQ_LEN * INPUT_DIM);
            let mut label_data = Vec::with_capacity(bs);
            for i in start_i..end_i {
                for t in 0..SEQ_LEN {
                    for d in 0..INPUT_DIM {
                        let raw = val_data[i].features[t * INPUT_DIM + d];
                        feat_data.push((raw - norm_means[d]) / norm_stds[d]);
                    }
                }
                label_data.push(val_data[i].label as i64);
            }

            let x = Tensor::<TrainBackend, 1>::from_floats(feat_data.as_slice(), &device)
                .reshape([bs, SEQ_LEN, INPUT_DIM]);
            let targets = Tensor::<TrainBackend, 1, Int>::from_ints(
                label_data
                    .iter()
                    .map(|&l| l as i32)
                    .collect::<Vec<_>>()
                    .as_slice(),
                &device,
            );

            let logits = model.forward(x);
            let loss =
                CrossEntropyLoss::new(None, &device).forward(logits.clone(), targets.clone());
            let lv: f32 = loss.into_data().to_vec().unwrap()[0];
            val_loss_sum += lv as f64 * bs as f64;

            let preds = logits.argmax(1);
            let correct: i64 = preds
                .equal(targets.unsqueeze_dim(1))
                .int()
                .sum()
                .into_scalar()
                .elem();
            val_correct += correct as usize;
            val_total += bs;
        }

        let tl = train_loss_sum / train_total as f64;
        let vl = val_loss_sum / val_total as f64;
        let ta = train_correct as f64 / train_total as f64;
        let va = val_correct as f64 / val_total as f64;
        let elapsed = start.elapsed().as_secs();

        eprintln!("Epoch {epoch}/{} | loss={tl:.4} acc={ta:.3} | val_loss={vl:.4} val_acc={va:.3} | {elapsed}s",
            args.epochs);

        if vl < best_val_loss {
            best_val_loss = vl;
            best_epoch = epoch;
            // Save best model
            let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
            model
                .clone()
                .save_file(out_dir.join("best_model"), &recorder)
                .expect("Save failed");
            eprintln!("  ✅ New best! val_loss={vl:.4} val_acc={va:.3}");
        }
    }

    eprintln!("\nDone! Best epoch={best_epoch} val_loss={best_val_loss:.4}");
    eprintln!("Total: {}s", start.elapsed().as_secs());
}
