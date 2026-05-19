# Hermes Agent Burn Tool

A public, sanitized Rust/Burn implementation of the custom Hermes/Joi agent inference and training tool.

This repo contains:

- `joi-burn` — WGPU/Metal/Vulkan inference binary for a temporal 300-brain swarm transformer
- `joi-train` — Burn autodiff training binary for SQLite-backed feature datasets
- SafeTensors loading helpers
- No private keys, API tokens, trading credentials, databases, logs, model dumps, or environment-specific config

## Status

This is a research tool. Treat outputs as experimental signals, not financial advice or autonomous trading instructions.

## Requirements

- Rust stable
- A WGPU-compatible device/backend
- macOS Metal or Linux Vulkan should work best

## Build

```bash
cargo build --release
```

## Inference

Put compatible SafeTensors weights in `weights/`:

```bash
mkdir -p weights
cp /path/to/swarm_v2_beast.safetensors weights/
./target/release/joi-burn
```

Model weights are excluded from git by default because they may be large or proprietary/private.

## Training

```bash
./target/release/joi-train \
  --db /path/to/features.sqlite \
  --epochs 40 \
  --batch-size 16 \
  --output ./burn_models_v3
```

Training data/database files are excluded from git. Use your own feature database with the schema expected by `src/train.rs`.

## Security / privacy

Before publishing, this repo was sanitized to exclude:

- `.env` files
- API keys/tokens/private keys
- local hostnames/IPs and personal absolute paths
- SQLite databases and JSONL logs
- model checkpoints/weights
- build artifacts

Run your own scan before redistributing:

```bash
grep -RInE 'api[_-]?key|secret|token|private|password|Bearer|sk-[A-Za-z0-9]|/Users/|10\.0\.|\.openclaw' . --exclude-dir target --exclude Cargo.lock
```

## License

MIT — see `LICENSE`.
