# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `70.00%` (7/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `2271615_0_crop_12793.jpg` | cut | cut | cut: 97.43% | danger: 2.26% | excluded: 0.31% | ✅ |
| 2 | `5_1249405_crop_6708.jpg` | danger | danger | cut: 1.27% | danger: 97.57% | excluded: 1.16% | ✅ |
| 3 | `2140097_5_crop_11751.jpg` | danger | danger | cut: 16.65% | danger: 67.83% | excluded: 15.53% | ✅ |
| 4 | `2_6_1749454_original_crop_4178.jpg` | excluded | excluded | cut: 5.17% | danger: 15.62% | excluded: 79.21% | ✅ |
| 5 | `1993137_0_crop_10361.jpg` | excluded | cut | cut: 56.21% | danger: 8.60% | excluded: 35.19% | ❌ |
| 6 | `2323851_0_crop_13688.jpg` | danger | danger | cut: 12.71% | danger: 81.77% | excluded: 5.53% | ✅ |
| 7 | `Yein_2022101101607_crop_17740.jpg` | danger | cut | cut: 62.09% | danger: 37.34% | excluded: 0.57% | ❌ |
| 8 | `1960307_0_crop_9363.jpg` | cut | danger | cut: 10.06% | danger: 89.61% | excluded: 0.33% | ❌ |
| 9 | `0_5199657_crop_849.jpg` | excluded | excluded | cut: 12.35% | danger: 0.76% | excluded: 86.89% | ✅ |
| 10 | `Bookwang_2022062802377_crop_14643.jpg` | danger | danger | cut: 3.01% | danger: 96.70% | excluded: 0.29% | ✅ |
