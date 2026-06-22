# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `team_share_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_team_share_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `70.00%` (7/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00409_FP_review_real_fp.jpg` | cut | danger | cut: 7.89% | danger: 92.03% | excluded: 0.08% | ❌ |
| 2 | `00178_original_GT_obj170.jpg` | danger | danger | cut: 0.46% | danger: 99.50% | excluded: 0.04% | ✅ |
| 3 | `00223_original_GT_duplicate_obj230.jpg` | danger | danger | cut: 0.68% | danger: 72.19% | excluded: 27.12% | ✅ |
| 4 | `00422_FP_review.jpg` | cut | danger | cut: 31.17% | danger: 68.53% | excluded: 0.30% | ❌ |
| 5 | `00008_original_GT_duplicate_obj5.jpg` | danger | cut | cut: 57.33% | danger: 42.54% | excluded: 0.13% | ❌ |
| 6 | `00222_original_GT_obj230.jpg` | danger | danger | cut: 0.38% | danger: 95.32% | excluded: 4.30% | ✅ |
| 7 | `00338_xlsx_excluded.jpg` | cut | cut | cut: 80.55% | danger: 19.23% | excluded: 0.23% | ✅ |
| 8 | `00309_xlsx_excluded.jpg` | excluded | excluded | cut: 4.61% | danger: 27.24% | excluded: 68.15% | ✅ |
| 9 | `00238_original_GT_duplicate_obj246.jpg` | danger | danger | cut: 10.00% | danger: 89.97% | excluded: 0.03% | ✅ |
| 10 | `00082_original_GT_duplicate_obj77.jpg` | danger | danger | cut: 17.65% | danger: 77.43% | excluded: 4.91% | ✅ |
