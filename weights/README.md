# Weights

Model weight files are intentionally not committed to this public repo.

Place a compatible SafeTensors file here, for example:

```bash
mkdir -p weights
cp /path/to/swarm_v2_beast.safetensors weights/
```

The default inference binary expects `weights/swarm_v2_beast.safetensors` unless you modify the source or pass your own path if your fork adds CLI config.
