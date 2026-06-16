# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `10.00%` (1/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00167_original_GT_obj161.jpg` | danger | cut | cut: 46.29% | danger: 41.56% | excluded: 12.15% | ❌ |
| 2 | `00003_original_GT_obj5.jpg` | danger | cut | cut: 57.16% | danger: 39.58% | excluded: 3.26% | ❌ |
| 3 | `00069_original_GT_duplicate_obj70.jpg` | danger | cut | cut: 65.38% | danger: 28.28% | excluded: 6.33% | ❌ |
| 4 | `00068_original_GT_duplicate_obj70.jpg` | danger | cut | cut: 68.58% | danger: 27.05% | excluded: 4.37% | ❌ |
| 5 | `00203_original_GT_obj198.jpg` | danger | cut | cut: 51.51% | danger: 45.47% | excluded: 3.02% | ❌ |
| 6 | `00050_original_GT_obj59.jpg` | danger | cut | cut: 48.31% | danger: 44.29% | excluded: 7.39% | ❌ |
| 7 | `00220_original_GT_obj227.jpg` | danger | danger | cut: 44.61% | danger: 49.49% | excluded: 5.90% | ✅ |
| 8 | `00182_original_GT_obj176.jpg` | danger | cut | cut: 60.24% | danger: 34.36% | excluded: 5.40% | ❌ |
| 9 | `00323_xlsx_excluded.jpg` | excluded | cut | cut: 65.89% | danger: 31.23% | excluded: 2.88% | ❌ |
| 10 | `00152_original_GT_duplicate_obj147.jpg` | danger | cut | cut: 61.20% | danger: 33.81% | excluded: 4.98% | ❌ |
