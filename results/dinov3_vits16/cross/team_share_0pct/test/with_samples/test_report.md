# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `team_share_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_team_share_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `60.00%` (6/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00036_original_GT_obj35.jpg` | cut | cut | cut: 67.63% | danger: 30.57% | excluded: 1.80% | ✅ |
| 2 | `00118_original_GT_obj117.jpg` | danger | cut | cut: 74.15% | danger: 24.57% | excluded: 1.27% | ❌ |
| 3 | `00165_original_GT_duplicate_obj155.jpg` | danger | danger | cut: 0.21% | danger: 99.68% | excluded: 0.11% | ✅ |
| 4 | `00172_original_GT_duplicate_obj164.jpg` | danger | danger | cut: 3.17% | danger: 96.79% | excluded: 0.04% | ✅ |
| 5 | `00149_original_GT_obj147.jpg` | danger | cut | cut: 96.54% | danger: 2.21% | excluded: 1.25% | ❌ |
| 6 | `00255_original_GT_duplicate_obj261.jpg` | danger | excluded | cut: 26.24% | danger: 14.56% | excluded: 59.20% | ❌ |
| 7 | `00143_original_GT_obj142.jpg` | danger | danger | cut: 3.58% | danger: 96.29% | excluded: 0.13% | ✅ |
| 8 | `00117_original_GT_obj116.jpg` | danger | cut | cut: 87.77% | danger: 11.44% | excluded: 0.80% | ❌ |
| 9 | `00246_original_GT_obj251.jpg` | danger | danger | cut: 2.31% | danger: 97.65% | excluded: 0.04% | ✅ |
| 10 | `00258_original_GT_duplicate_obj263.jpg` | danger | danger | cut: 4.67% | danger: 93.68% | excluded: 1.65% | ✅ |
