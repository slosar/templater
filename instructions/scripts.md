# Training Notes

Representative commands used in this repo:

```bash
# Baseline free-form templates
python scripts/train.py \
  --n-spectra 5000 --zmin 0.4 --zmax 1.1 \
  --Nt 5 --Nz 200 --n-epochs 20

# Blind-z curriculum
python scripts/train.py \
  --n-spectra 5000 --zmin 0.4 --zmax 1.1 \
  --Nt 5 --Nz 200 --n-epochs 20 \
  --disable-z-prior 0.99 --z-prior-warmup 30

# Neural template model
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python scripts/train.py \
  --neural-templates \
  --Nt 4 --n-spectra 50000 \
  --zmin 0.75 --zmax 1.0 \
  --zmin-loader 0.75 --zmax-loader 1.0 \
  --Nz 1000 --n-epochs 200 --batch-size 8192 \
  --disable-z-prior 0.00 --z-prior-warmup 30 \
  --nz-steps 0 --nz-sigma 0.6
```

Notes:

- `spectra_shuffled/` is the default training input because it improves sequential I/O.
- `--resume-templates-only` is the path for reusing learned templates with a fresh `n(z)`.
- `--target-noise` is for controlled SNR degradation experiments and should match between training and evaluation if used.
