# Vectorialization

This document is an alias-style note for the vectorization stage. In this repository, the implementation and primary documentation use the term **vectorization**.

See [vectorization.md](vectorization.md) for B1.7 details.

B1.8 adds the next step: preprocessing. After `X.npy` and optional `y.npy` are produced under `processed_data_dir/vectorized`, run:

```bash
python scripts/preprocess_collide.py --config configs/paths/local.yaml
```

This writes standardized arrays and metadata under `processed_data_dir/preprocessed`, preparing the tiny local/HF sample for the next milestone: minimal classification training.

