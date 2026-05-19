# Hermes Agent Burn Tool

A public, sanitized Rust/Burn implementation of the custom **Hermes/Joi agent** inference and training tool.

This repo is meant for people running **Hermes** or **OpenClaw** agent stacks who want a local Burn/WGPU signal model they can wire into their own agent runtime.

This repo contains:

- `joi-burn` — WGPU/Metal/Vulkan inference binary for a temporal 300-brain swarm transformer
- `joi-train` — Burn autodiff training binary for SQLite-backed feature datasets
- SafeTensors loading helpers
- Example input/metadata files
- No private keys, API tokens, trading credentials, databases, logs, model dumps, or environment-specific config

## Status

This is a research tool. Treat outputs as experimental signals, not financial advice or autonomous trading instructions.

## Requirements

- Rust stable
- A WGPU-compatible device/backend
- macOS Metal or Linux Vulkan should work best
- Optional: Hermes or OpenClaw if you want to run it as an agent-side tool

## Build

```bash
cargo build --release
```

The release binaries will be:

```bash
./target/release/joi-burn
./target/release/joi-train
```

## Hermes setup

Use this repo as a local Hermes tool directory or clone it into your Hermes workspace:

```bash
git clone https://github.com/ivanontech/hermes-agent-burn.git
cd hermes-agent-burn
cargo build --release
```

Example Hermes-style tool command:

```bash
./target/release/joi-burn \
  --model-dir ./weights \
  --input ./examples/input_sequence.json \
  --output ./burn_brain_signal.json
```

Example Hermes training command:

```bash
./target/release/joi-train \
  --db ./data/features.sqlite \
  --epochs 40 \
  --batch-size 16 \
  --output ./burn_models_v3
```

Suggested Hermes env/config values:

```bash
HERMES_BURN_MODEL_DIR=/path/to/hermes-agent-burn/weights
HERMES_BURN_INPUT=/path/to/input_sequence.json
HERMES_BURN_OUTPUT=/path/to/burn_brain_signal.json
```

Keep these values local in your own Hermes config or `.env`. Do **not** commit private model paths, live signal outputs, databases, or secrets.

## OpenClaw setup

OpenClaw users can run the same binaries from a workspace task, cron, or agent tool wrapper:

```bash
git clone https://github.com/ivanontech/hermes-agent-burn.git
cd hermes-agent-burn
cargo build --release
./target/release/joi-burn --model-dir ./weights --input ./examples/input_sequence.json
```

If you wire this into OpenClaw memory/workspace paths, keep local paths in your own config and do not commit them back to the repo.

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
grep -RInE 'api[_-]?key|secret|token|private|password|Bearer|sk-[A-Za-z0-9]|/Users/|10\.0\.|\.openclaw|\.hermes' . --exclude-dir target --exclude Cargo.lock
```

## License

MIT — see `LICENSE`.
