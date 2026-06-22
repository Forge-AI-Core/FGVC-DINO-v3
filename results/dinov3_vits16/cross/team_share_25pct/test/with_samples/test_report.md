# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `team_share_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_team_share_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `80.00%` (8/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00211_original_GT_obj213.jpg` | cut | cut | cut: 59.23% | danger: 35.80% | excluded: 4.97% | ✅ |
| 2 | `00246_original_GT_obj251.jpg` | danger | danger | cut: 17.75% | danger: 82.04% | excluded: 0.21% | ✅ |
| 3 | `00120_original_GT_duplicate_obj121.jpg` | danger | danger | cut: 3.32% | danger: 96.30% | excluded: 0.38% | ✅ |
| 4 | `00181_original_GT_obj173.jpg` | danger | danger | cut: 13.04% | danger: 86.82% | excluded: 0.14% | ✅ |
| 5 | `00270_original_GT_obj272.jpg` | danger | excluded | cut: 7.86% | danger: 30.62% | excluded: 61.52% | ❌ |
| 6 | `00386_FP_review_real_fp.jpg` | excluded | danger | cut: 3.62% | danger: 93.15% | excluded: 3.24% | ❌ |
| 7 | `00109_original_GT_duplicate_obj109.jpg` | danger | danger | cut: 5.42% | danger: 90.69% | excluded: 3.90% | ✅ |
| 8 | `00237_original_GT_duplicate_obj246.jpg` | danger | danger | cut: 7.47% | danger: 92.34% | excluded: 0.19% | ✅ |
| 9 | `00333_xlsx_excluded.jpg` | cut | cut | cut: 61.05% | danger: 37.83% | excluded: 1.12% | ✅ |
| 10 | `00292_hidden_GT_from_FP_review_obj90019.jpg` | danger | danger | cut: 5.26% | danger: 92.47% | excluded: 2.27% | ✅ |
