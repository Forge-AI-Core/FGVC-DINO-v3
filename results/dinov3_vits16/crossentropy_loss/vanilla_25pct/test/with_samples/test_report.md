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
| 1 | `00409_FP_review_real_fp.jpg` | cut | danger | cut: 22.84% | danger: 76.96% | excluded: 0.20% | ❌ |
| 2 | `00264_original_GT_duplicate_obj266.jpg` | danger | danger | cut: 44.99% | danger: 51.44% | excluded: 3.57% | ✅ |
| 3 | `00232_original_GT_obj238.jpg` | danger | danger | cut: 7.01% | danger: 57.79% | excluded: 35.20% | ✅ |
| 4 | `00096_original_GT_obj95.jpg` | danger | danger | cut: 3.80% | danger: 56.42% | excluded: 39.78% | ✅ |
| 5 | `00098_original_GT_obj98.jpg` | danger | excluded | cut: 39.71% | danger: 16.25% | excluded: 44.04% | ❌ |
| 6 | `00207_original_GT_obj208.jpg` | danger | danger | cut: 5.48% | danger: 94.28% | excluded: 0.24% | ✅ |
| 7 | `00245_original_GT_duplicate_obj250.jpg` | danger | cut | cut: 78.58% | danger: 12.70% | excluded: 8.72% | ❌ |
| 8 | `00430_FP_review.jpg` | excluded | danger | cut: 4.08% | danger: 93.32% | excluded: 2.60% | ❌ |
| 9 | `00272_original_GT_obj10003.jpg` | danger | cut | cut: 90.15% | danger: 9.58% | excluded: 0.27% | ❌ |
| 10 | `00305_xlsx_excluded.jpg` | cut | cut | cut: 78.20% | danger: 7.99% | excluded: 13.81% | ✅ |
