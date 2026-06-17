from pathlib import Path
import SimpleITK as sitk

input_dir = Path("/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MICCAI_AIMS_TBI")
output_dir = Path("/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/OUTPUT_MHA")
output_dir.mkdir(parents=True, exist_ok=True)

for nifti_path in input_dir.glob("*T1.nii.gz"):
    print(f"Converting {nifti_path.name}")

    img = sitk.ReadImage(str(nifti_path))

    mha_name = nifti_path.name.replace(".nii.gz", ".mha")
    mha_path = output_dir / mha_name

    sitk.WriteImage(img, str(mha_path))

print("Done.")
# import SimpleITK as sitk

# img = sitk.ReadImage("/data/data/DA25S005/miccai_tbi/MultiTalentV2_finetuning/MICCAI_AIMS_TBI/scan_0001_T1.nii.gz")

# for k in img.GetMetaDataKeys():
#     print(k, img.GetMetaData(k))