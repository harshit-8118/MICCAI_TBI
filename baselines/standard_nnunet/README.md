# Standard nnU-Net baseline against Model B

This folder contains every script and configuration for the **standard
nnU-Net baseline**. This workflow creates a new **standard non-residual nnU-Net v2** baseline. It
does not modify, replace, retrain, or ensemble the submitted Model A/B models.
The comparison is retrospective development evidence only: Phase-2 validation
has been used for development and is not an independent holdout.

## Fixed protocol

- Comparator: historical Model B (`e5ohnz5w`), documented in
  `model_b_reference.yaml`.
- Training data: the exact 522-case development pool and fold-0 90/10 split
  from Model B's original 10-fold split JSON.
- Excluded cases: the same 30 records excluded before Model B training. They
  must not enter `imagesTr`, `labelsTr`, or `splits_final.json`.
- Baseline method: official nnU-Net v2 `nnUNetTrainer`, `3d_fullres`, T1 only,
  and no MultiTalentV2 checkpoint.
- No released-validation tuning: evaluate the new baseline only at the two
  pre-specified operating points below.

The baseline answers whether Model B exceeds a conventional nnU-Net. It does
not isolate the value of pretraining because Model B also uses a residual
architecture. A later random-initialized residual control is required for that
separate causal question.

## Why its training differs from Model B

This is deliberately a **standard nnU-Net baseline**, not a random-init Model
B clone. It matches Model B's input modality, 522-case development pool,
fold-0 partition, and external-evaluation protocol. It does not match its
architecture, optimizer, learning rate, or staged fine-tuning schedule.

Model B starts from a MultiTalentV2 checkpoint, so its small AdamW learning
rate and staged controls are transfer-learning choices intended to preserve
pretrained representations. The baseline starts from random He/Kaiming
initialization and follows the canonical `nnUNetTrainer`: plain-convolutional
U-Net, SGD (LR 0.01, momentum 0.99, Nesterov), 1,000 epochs with polynomial
decay, deep supervision, and no head/partial/full stages or warm-up. Do not
substitute Model B's AdamW LR of `1e-4`, its weighted loss, or its staging into
this baseline; that would no longer be a conventional nnU-Net comparison.

nnU-Net's exact 3-D patch size, batch size, number of stages, feature widths,
and consequently parameter count are generated from the dataset fingerprint
by the default 8-GB reference planner. The A6000 runs the job but does not
justify retuning the planner. Record the resulting plan and parameter count;
do not claim a fixed parameter count beforehand.

## Preparation on the training machine

Choose an unused dataset ID if `501` is already occupied. Set the three
nnU-Net environment variables to persistent locations with sufficient space.
Run the commands in an environment that contains the packages listed in
`requirements.txt` and a pinned nnU-Net v2 installation.

```bash
export nnUNet_raw=/data/nnunet_raw
export nnUNet_preprocessed=/data/nnunet_preprocessed
export nnUNet_results=/data/nnunet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

python3 baselines/standard_nnunet/prepare_standard_nnunet_baseline.py \
  --config baselines/standard_nnunet/model_b_matched_baseline.yaml \
  --source-dataset-dir /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MICCAI_AIMS_TBI \
  --model-b-split-file /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/checkpoints/train_val_test_10fold_seed42.json \
  --raw-root "$nnUNet_raw" \
  --link-mode hardlink
```

The command fails rather than silently continuing if it cannot verify all of:
552 local cases, a 522-case development pool, 30 excluded cases, 10 folds, and
the exact selected fold-0 partition. Before it creates any nnU-Net files, it
fully decompresses and validates every local T1/lesion pair (shape, affine,
finite voxels, binary lesion labels). It does not read, stage, link, copy, plan
on, preprocess, or train on any released-validation case. It writes a
source-split hash and a case manifest to
`checkpoints/paper_baselines/standard_nnunet_model_b_matched/`.

## Official nnU-Net v2 commands

The preparation script records the installed nnU-Net version in the baseline
manifest. Use the standard planner/trainer; do not import MultiTalentV2 weights
or use a custom residual trainer.

```bash
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity

python3 baselines/standard_nnunet/report_standard_nnunet_architecture.py \
  --plans "$nnUNet_preprocessed/Dataset501_AIMSTBI_T1_Standard/nnUNetPlans.json" \
  --dataset-json "$nnUNet_raw/Dataset501_AIMSTBI_T1_Standard/dataset.json" \
  --reference-plans /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MultiTalentV2_pretrained/Dataset617_nativect/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/nnUNetResEncUNetL1x1x1_Plans_znorm_bs24.json \
  --reference-deep-supervision false \
  --output checkpoints/paper_baselines/standard_nnunet_model_b_matched/architecture_report.json

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 0 --npz -device cuda
```

The report distinguishes the upstream MultiTalentV2 pretraining plan from
historical Model-B fine-tuning. The upstream plan's `192^3`/batch-24 values
must never be described as Model B settings: Model B used `160^3` patches and
batch size 2.

## W&B and size-stratified reporting

The current nnU-Net v2 `MetaLogger` has built-in W&B support. Enable it with
environment variables; this is logging only and does **not** change the
trainer, optimizer, loss, checkpointing, or predictions.

```bash
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project=AIMS-TBI-standard-nnUNet
export nnUNet_wandb_mode=online
export WANDB_ENTITY=da25s003-indian-institute-of-technology-madras  # if needed

CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 501 3d_fullres 0 --npz -device cuda
```

The native run logs training loss, validation-patch loss, foreground Dice,
EMA Dice, and learning rate per epoch. Those validation values are sampled
patch statistics, not case-level outcomes. Do **not** report stratified patch
Dice in the paper or use it to choose a checkpoint.

After full-volume released-validation inference, the fixed evaluator writes
case-level and summary metrics for `empty`, `micro` (1--999 lesion voxels),
`small` (1,000--4,999), and `large` (>=5,000), as well as all/positive/gt50.
The CSV keys are deliberately named `released_validation_*`, not `test_*`.
Upload those fixed, post-training summaries to W&B as a linked analysis run:

```bash
python3 baselines/standard_nnunet/log_standard_nnunet_summary_to_wandb.py \
  --summary checkpoints/paper_baselines/standard_nnunet_model_b_matched/phase2_fixed_evaluation/summary.csv \
  --project AIMS-TBI-standard-nnUNet \
  --name standard_nnunet_fold0_released_validation_fixed \
  --mode online
```

This is safe for paper tracking because it is a fixed post-training report.
It remains development evidence, not a hidden-test result. In the manuscript,
label the table/figure “released validation (development)” and give the
case counts alongside every size stratum. Do not use the logged values to
replace the pre-registered `checkpoint_final.pth`.

The report records the actual planned nnU-Net capacity and, when the
MultiTalentV2 plan path is supplied, Model B's architecture and parameter
count using the same one-input/two-output definition. Include these measured
values in the methods table, rather than an estimate based only on patch size.

Use `checkpoint_final.pth`, specified in the baseline configuration, for the
pre-registered external inference. This avoids selecting a new checkpoint by
repeatedly inspecting released Phase-2 validation performance. Only after
training is complete, create a separate inference-only input folder. It is not
under `nnUNet_raw`, and labels are never copied there.

```bash
python3 baselines/standard_nnunet/prepare_released_validation_inference_input.py \
  --released-validation-dir /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/Validation2025_100/Validation2025_100 \
  --output-dir checkpoints/paper_baselines/standard_nnunet_model_b_matched/released_validation_inputs \
  --link-mode hardlink
```

```bash
nnUNetv2_predict \
  -i checkpoints/paper_baselines/standard_nnunet_model_b_matched/released_validation_inputs \
  -o checkpoints/paper_baselines/standard_nnunet_model_b_matched/phase2_predictions \
  -d 501 -c 3d_fullres -f 0 \
  -chk checkpoint_final.pth \
  --save_probabilities
```

This is full-volume sliding-window inference across the complete
preprocessed brain volume, not one fixed 160-cube inference. The standard
planner determines the cube size; the command uses the standard 0.5 tile step,
Gaussian overlap weighting, and mirror TTA. For a normal 3-D plan with three
mirror axes, that is eight predictions per sliding-window tile (the original
plus all seven flip combinations). Do not add `--disable_tta` to the primary
standard-nnU-Net command.

Keep the NIfTI masks and `.npz` probability files. The probabilities are needed
to apply the fixed `tau=0.20, minCC=40` primary operating point and the fixed
`tau=0.50, minCC=0` sensitivity operating point.

```bash
python3 baselines/standard_nnunet/evaluate_standard_nnunet_external.py \
  --dataset-dir /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/Validation2025_100/Validation2025_100 \
  --prediction-dir checkpoints/paper_baselines/standard_nnunet_model_b_matched/phase2_predictions \
  --output-dir checkpoints/paper_baselines/standard_nnunet_model_b_matched/phase2_fixed_evaluation \
  --thresholds 0.20 0.50 \
  --min-components 40 0
```

This evaluator uses the same canonical 1-mm grid and local empty-mask surface
policy as `evaluate_external_validation.py`, and writes per-case CSVs required
for bootstrap confidence intervals. Re-run Model B with
`evaluate_external_validation.py` at these same two fixed settings to obtain
the matched Model-B per-case CSVs.

## Paper wording

Describe this as a conventional external baseline trained on the same local
development partition as Model B. Do not describe it as an independent test or
as an isolated estimate of pretraining benefit.
