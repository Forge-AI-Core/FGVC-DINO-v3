# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `50.00%` (5/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00170_original_GT_duplicate_obj163.jpg` | danger | cut | cut: 70.29% | danger: 29.47% | excluded: 0.24% | ❌ |
| 2 | `00238_original_GT_duplicate_obj246.jpg` | danger | danger | cut: 5.22% | danger: 94.54% | excluded: 0.24% | ✅ |
| 3 | `00082_original_GT_duplicate_obj77.jpg` | danger | excluded | cut: 4.81% | danger: 4.06% | excluded: 91.13% | ❌ |
| 4 | `00230_original_GT_obj237.jpg` | danger | cut | cut: 82.51% | danger: 10.65% | excluded: 6.84% | ❌ |
| 5 | `00400_FP_review_real_fp.jpg` | cut | danger | cut: 5.20% | danger: 93.02% | excluded: 1.77% | ❌ |
| 6 | `00325_xlsx_excluded.jpg` | excluded | excluded | cut: 8.34% | danger: 1.85% | excluded: 89.80% | ✅ |
| 7 | `00167_original_GT_obj161.jpg` | danger | danger | cut: 1.81% | danger: 95.47% | excluded: 2.72% | ✅ |
| 8 | `00263_original_GT_obj266.jpg` | danger | cut | cut: 61.15% | danger: 7.32% | excluded: 31.53% | ❌ |
| 9 | `00261_original_GT_duplicate_obj265.jpg` | danger | danger | cut: 28.78% | danger: 70.78% | excluded: 0.44% | ✅ |
| 10 | `00220_original_GT_obj227.jpg` | danger | danger | cut: 29.47% | danger: 67.28% | excluded: 3.25% | ✅ |
