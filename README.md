# Hybrid Ensemble with Multi-Patch Fine-Tuning for Traumatic Brain Injury Segmentation

### [Checkpoints](https://doi.org/10.5281/zenodo.22784159)

This repository contains the code and analysis used for T1-weighted MRI lesion
segmentation in the AIMS-TBI 2026 challenge. The final system combines two
MultiTalentV2-initialized residual U-Nets by equal-weight probability averaging:

- **Model A**: large-to-small patch fine-tuning (192³ to 160³), used without TTA.
- **Model B**: independently fine-tuned at 160³, used with flip-based TTA.
- **Inference**: sliding-window probability averaging, thresholding, and
  3-D connected-component filtering.

The submitted configuration obtained lesion-positive Dice of **0.609** on the
released Phase-2 validation set and **0.5533** on the hidden challenge test set.

## Repository layout

- `multitalent_tbi/` — dataset loading, residual U-Net construction, training,
  inference, losses, post-processing, and split utilities.
- `train.py` and `train_hierarchical_segmenter.py` — fine-tuning entry points.
- `evaluate_external_validation.py` and `inference_sweep.py` — evaluation and
  fixed-setting inference analyses.
- `baselines/` — standard nnU-Net, matched randomly initialized residual U-Net,
  bootstrap confidence intervals, and paired statistical comparisons.

## Data and checkpoints

The AIMS-TBI images, reference masks, MultiTalentV2 pretrained weights, and
fine-tuned checkpoints are not distributed in this repository. Configure their
local paths in a YAML configuration before training or inference. The released
Phase-2 validation data are used only for locked post-training evaluation.

## Reproducing analyses

Install the dependencies listed in `requirements.txt`, provide a local YAML
configuration and checkpoint paths, then use:

```bash
python train.py --config path/to/config.yaml
python evaluate_external_validation.py --help
```

Baseline-specific preparation and commands are documented in
`baselines/standard_nnunet/README.md` and
`baselines/random_residual_nnunet/README.md`.

## Notes on interpretation

The revised comparison evaluates all configurations at common operating points.
Reported bootstrap intervals quantify uncertainty across cases for a single
training seed; they do not quantify variability across independent retraining
runs.

## Citation

Citation details will be added after the workshop proceedings are released.
