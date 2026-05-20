# Hermes Agent Burn Tool

**Rust/Burn local inference + training for agent-side temporal signal models.**

This is a public, sanitized implementation of a Hermes/OpenClaw-adjacent Burn tool: a GPU-capable Rust binary for running local sequence models beside an agent runtime, plus a training binary for SQLite-backed feature datasets.

The short version: Hermes should not need to call a giant remote LLM for every local signal. This repo shows the lower-level path — small, fast, inspectable models running locally through Burn/WGPU and feeding structured outputs back into an agent loop.

## What ships

- `joi-burn` — WGPU/Metal/Vulkan inference binary for temporal multi-brain/swarm-style signal models.
- `joi-train` — Burn autodiff training binary for SQLite feature datasets.
- SafeTensors loading helpers.
- Example input and metadata files.
- Public-safe docs and sample commands.
- No private keys, API tokens, trading credentials, local databases, logs, or environment-specific config.

## Architecture

```text
Hermes/OpenClaw task
      │
      ▼
feature sequence JSON / local feature DB
      │
      ├── joi-burn  ──► SafeTensors weights ──► structured signal JSON
      │
      └── joi-train ──► SQLite features ───────► new Burn model artifacts
```

The intended integration pattern is simple: keep the agent orchestration in Hermes, keep heavy/local numeric inference in Rust/Burn, and pass only structured JSON across the boundary.

## Why this is interesting for agent infrastructure

- **Local-first inference:** runs on WGPU-compatible devices instead of requiring a hosted model endpoint.
- **Agent-friendly IO:** JSON in, JSON out, easy to wrap as a Hermes tool or scheduled worker.
- **Burn ecosystem:** demonstrates Rust-native model loading/training paths that can sit next to production infrastructure.
- **Security boundary:** local paths, private datasets, wallets, and credentials stay outside the public repo.
- **Composable:** can be used as a scoring primitive for trading signals, routing, health models, or any time-series agent heuristic.

## Status

Research/experimental. Outputs are model signals, not financial advice and not autonomous trading instructions.

## Requirements

- Rust stable
- WGPU-compatible backend
- macOS Metal or Linux Vulkan recommended
- Optional: Hermes Agent or OpenClaw if you want to wire it into an agent runtime

## Build

```bash
git clone https://github.com/ivanontech/hermes-agent-burn.git
cd hermes-agent-burn
cargo build --release
```

Release binaries:

```text
./target/release/joi-burn
./target/release/joi-train
```

## Inference quickstart

```bash
./target/release/joi-burn \
  --model-dir ./weights \
  --input ./examples/input_sequence.json \
  --output ./burn_brain_signal.json
```

Expected integration contract:

```text
input_sequence.json  →  joi-burn  →  burn_brain_signal.json
```

Keep output JSON local if it contains live strategy state.

## Training quickstart

```bash
./target/release/joi-train \
  --db ./data/features.sqlite \
  --epochs 40 \
  --batch-size 16 \
  --output ./burn_models_v3
```

Training data/database files are intentionally excluded from Git. Use your own feature database with the schema expected by `src/train.rs`.

## Hermes setup

Clone and build the repo inside your workspace:

```bash
git clone https://github.com/ivanontech/hermes-agent-burn.git
cd hermes-agent-burn
cargo build --release
```

Example local-only environment values:

```bash
HERMES_BURN_MODEL_DIR=/path/to/hermes-agent-burn/weights
HERMES_BURN_INPUT=/path/to/input_sequence.json
HERMES_BURN_OUTPUT=/path/to/burn_brain_signal.json
```

Keep those values in your own Hermes config or `.env`. Do **not** commit private paths, live signal outputs, databases, wallets, or secrets.

## Files

```text
src/main.rs                  inference binary
src/train.rs                 training binary
examples/input_sequence.json sample feature sequence
weights/README.md            weight placement notes
Cargo.toml                   Burn/WGPU dependencies
```

## Verification

```bash
cargo fmt --check
cargo check
```

For release verification:

```bash
cargo build --release
./target/release/joi-burn --help
./target/release/joi-train --help
```

## Security / privacy

Before publishing, this repo was sanitized to exclude:

- `.env` files;
- API keys/tokens/private keys;
- wallet material;
- local hostnames/IPs and personal absolute paths;
- SQLite databases and JSONL logs;
- private model checkpoints/weights;
- build artifacts.

Run your own scan before redistributing. Report pattern names and filenames only — never paste matched secret values into chat/logs.

```bash
grep -RInE 'api[_-]?key|secret|token|private|password|Bearer|sk-[A-Za-z0-9]|/Users/|10\.0\.|\.openclaw|\.hermes' . \
  --exclude-dir target \
  --exclude Cargo.lock
```

## Related

- [`ivanontech/skunkworks`](https://github.com/ivanontech/skunkworks) — AI ops dashboard and Alpha Radar cockpit.
- [`ivanontech/hermes-burn-tool-router`](https://github.com/ivanontech/hermes-burn-tool-router) — tiny Burn neural router for Hermes tool/control classification.

## License

MIT — see `LICENSE`.
