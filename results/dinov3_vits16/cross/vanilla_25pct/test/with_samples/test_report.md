# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `50.00%` (5/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00223_original_GT_duplicate_obj230.jpg` | danger | excluded | cut: 3.46% | danger: 3.35% | excluded: 93.19% | ❌ |
| 2 | `00120_original_GT_duplicate_obj121.jpg` | danger | danger | cut: 20.13% | danger: 79.23% | excluded: 0.63% | ✅ |
| 3 | `00067_original_GT_obj70.jpg` | danger | danger | cut: 17.88% | danger: 80.35% | excluded: 1.76% | ✅ |
| 4 | `00233_original_GT_obj240.jpg` | danger | danger | cut: 27.11% | danger: 72.37% | excluded: 0.51% | ✅ |
| 5 | `00142_original_GT_obj140.jpg` | danger | cut | cut: 54.44% | danger: 21.05% | excluded: 24.52% | ❌ |
| 6 | `00065_original_GT_obj68.jpg` | danger | excluded | cut: 1.06% | danger: 3.48% | excluded: 95.46% | ❌ |
| 7 | `00333_xlsx_excluded.jpg` | cut | cut | cut: 59.05% | danger: 26.72% | excluded: 14.23% | ✅ |
| 8 | `00074_original_GT_obj73.jpg` | danger | cut | cut: 72.71% | danger: 26.98% | excluded: 0.32% | ❌ |
| 9 | `00072_original_GT_duplicate_obj71.jpg` | danger | cut | cut: 97.61% | danger: 1.14% | excluded: 1.25% | ❌ |
| 10 | `00398_FP_review_real_fp.jpg` | cut | cut | cut: 86.01% | danger: 13.30% | excluded: 0.69% | ✅ |
