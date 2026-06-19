# Step 1 — build model tarball (run once, ~30 seconds)
bash build_model_tarball.sh \
  --checkpoint /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/checkpoints/trained_models/best_tr_f1_kpcyjb66.pt \
  --plans /data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MultiTalentV2_pretrained/Dataset617_nativect/MultiTalent_trainer_4000ep__nnUNetResEncUNetL1x1x1_Plans_znorm_bs24__3d_fullres/fold_all/nnUNetResEncUNetL1x1x1_Plans_znorm_bs24.json

# Step 2 — convert a test scan to .mha (one-liner)
python3 -c "
import SimpleITK as sitk
img = sitk.ReadImage('/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MICCAI_AIMS_TBI/scan_0001_T1.nii.gz')
sitk.WriteImage(img, 'test/input/images/t1-brain-mri/T1w.mha')
print('Converted OK')
"

# Step 3 — extract model for local test
mkdir -p test/model && tar -xzvf algorithmmodel.tar.gz -C test/model/

# Step 4 — build Docker image (~5-10 min first time)
bash save.sh

# Step 5 — run local test
bash test_run.sh


```mermaid
flowchart TD
    A["Training Data<br/>T1 MRI + Lesion Masks"] --> B1["Lesion-Positive Cases"]
    A --> B2["Empty Cases<br/>Held Aside"]

    B1 --> C["Segmenter A<br/>MultiTalentV2 Fine-Tuning<br/>Lesion-Only Training"]
    C --> D["Tiny-Lesion Focus<br/>Heavy tiny oversampling<br/>Foreground patch sampling<br/>Lesion-sensitive loss"]

    D --> E["Run Segmenter A<br/>on Empty MRIs"]
    B2 --> E

    E --> F["Hard Negatives<br/>False-positive patches<br/>Artifacts, CSF edges, metal clips,<br/>WM shadows, brain edge errors"]

    F --> G["Hard-Negative Fine-Tuning<br/>Mostly lesion-positive batches<br/>+ 10-20% empty/hard-negative patches"]

    G --> H["3-5 Fold / Seed Models<br/>Selected by lesion-only Dice<br/>not all-case Dice"]

    H --> I["Inference Ensemble<br/>Sequential load/unload<br/>T4 16GB safe"]

    I --> J["TTA<br/>Original + X/Y/Z flips<br/>Average probabilities after unflip"]

    J --> K["Candidate Components<br/>Connected components from<br/>ensemble probability mask"]

    K --> L["Component Features<br/>Size, mean/max probability<br/>model votes, entropy<br/>TTA stability, intensity contrast<br/>location priors"]

    L --> M["Second-Stage Calibrator<br/>Keep/reject each component"]

    M --> N1["Segmentation Submission<br/>High recall<br/>remove only obvious junk"]
    M --> N2["Detection Submission<br/>Stricter empty-vs-lesion decision"]

    N1 --> O["Goal<br/>Better lesion-only Dice<br/>especially tiny/small lesions"]
    N2 --> P["Goal<br/>Better empty/lesion classification"]
```

python3 inference_sweep.py   --config config.yml   --fold 1   --split val   --positive-only   --load-mode preload   --checkpoints     checkpoints/trained_models/best_tr_f1_kpcyjb66.pt     checkpoints/tbi_multitalentv2/segmenter_a/fold_1/best_tiny.pt     checkpoints/tbi_multitalentv2/segmenter_a/fold_1/best_gt50.pt checkpoints/trained_models/best_714109.pt  --thresholds 0.10 0.15 0.20 0.25   --min-components 0 3 5   --output-dir checkpoints/tbi_multitalentv2/inference_sweeps/fast_ensemble_f1