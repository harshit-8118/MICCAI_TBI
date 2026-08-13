# Matched random-initialized residual nnU-Net control

This is the reviewer-required scratch control for Model B. It keeps Model B's
residual `ResidualEncoderUNet` architecture and effective training protocol,
but starts from a seeded Kaiming-normal initialization rather than loading
MultiTalentV2. It is therefore the experiment that can isolate initialization,
not the conventional plain nnU-Net baseline.

The training configuration uses only the existing 522-case development pool
and fold 0 from `train_val_test_10fold_seed42.json`. The 30 excluded local
cases remain outside fold 0. No released Phase-2 paths are present in the
configuration; do not access that dataset until training finishes and the final
checkpoint is fixed.

Prepare and audit before launching:

```bash
python3 baselines/random_residual_nnunet/prepare_random_residual_baseline.py \
  --config baselines/random_residual_nnunet/model_b_matched_random_init.yaml
```

After the current standard nnU-Net GPU job is complete:

```bash
CUDA_VISIBLE_DEVICES=0 python3 baselines/random_residual_nnunet/train_random_residual_baseline.py \
  --config baselines/random_residual_nnunet/model_b_matched_random_init.yaml \
  --fold 0
```

The new output folder must contain `initialization_audit.json` with
`load_pretrained: false`, `init_checkpoint: null`, `resume_checkpoint: null`,
and a deterministic initial-state SHA-256. Do not use `--init-checkpoint`,
`--resume-checkpoint`, or `--allow-existing-output` for the first run.
