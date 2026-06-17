# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_50pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_50pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `60.00%` (6/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00274_hidden_GT_from_FP_review_obj90001.jpg` | danger | danger | cut: 19.02% | danger: 80.72% | excluded: 0.26% | ✅ |
| 2 | `00238_original_GT_duplicate_obj246.jpg` | danger | danger | cut: 8.07% | danger: 91.75% | excluded: 0.18% | ✅ |
| 3 | `00416_FP_review_real_fp.jpg` | cut | cut | cut: 62.04% | danger: 37.79% | excluded: 0.17% | ✅ |
| 4 | `00404_FP_review_real_fp.jpg` | excluded | danger | cut: 5.96% | danger: 93.81% | excluded: 0.23% | ❌ |
| 5 | `00067_original_GT_obj70.jpg` | danger | danger | cut: 40.53% | danger: 59.26% | excluded: 0.20% | ✅ |
| 6 | `00032_original_GT_obj32.jpg` | danger | excluded | cut: 3.90% | danger: 0.80% | excluded: 95.30% | ❌ |
| 7 | `00007_original_GT_duplicate_obj5.jpg` | danger | cut | cut: 98.15% | danger: 1.73% | excluded: 0.12% | ❌ |
| 8 | `00186_original_GT_duplicate_obj178.jpg` | danger | excluded | cut: 41.59% | danger: 4.81% | excluded: 53.60% | ❌ |
| 9 | `00185_original_GT_duplicate_obj178.jpg` | danger | danger | cut: 38.09% | danger: 59.97% | excluded: 1.94% | ✅ |
| 10 | `00053_original_GT_duplicate_obj59.jpg` | danger | danger | cut: 2.52% | danger: 97.21% | excluded: 0.27% | ✅ |
