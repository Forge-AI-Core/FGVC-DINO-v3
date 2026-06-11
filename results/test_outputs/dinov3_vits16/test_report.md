# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `40.00%` (4/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `3_5201507_crop_5434.jpg` | excluded | cut | cut: 59.45% | danger: 34.50% | excluded: 6.05% | ❌ |
| 2 | `0_5204511_crop_1082.jpg` | excluded | cut | cut: 84.94% | danger: 8.97% | excluded: 6.09% | ❌ |
| 3 | `2_5200881_crop_4495.jpg` | excluded | cut | cut: 39.60% | danger: 25.00% | excluded: 35.40% | ❌ |
| 4 | `2181738_3_crop_11899.jpg` | cut | cut | cut: 79.29% | danger: 17.16% | excluded: 3.55% | ✅ |
| 5 | `Hyunil_2022011802281_crop_15373.jpg` | danger | cut | cut: 56.72% | danger: 38.86% | excluded: 4.42% | ❌ |
| 6 | `Yein_2022101101609_crop_17759.jpg` | cut | cut | cut: 85.99% | danger: 11.59% | excluded: 2.42% | ✅ |
| 7 | `4_0_985806_crop_5793.jpg` | cut | cut | cut: 81.70% | danger: 14.22% | excluded: 4.09% | ✅ |
| 8 | `Jeongwon_2022091702876_crop_16135.jpg` | cut | cut | cut: 75.67% | danger: 21.19% | excluded: 3.14% | ✅ |
| 9 | `0_5205740_crop_1201.jpg` | danger | cut | cut: 50.34% | danger: 38.55% | excluded: 11.11% | ❌ |
| 10 | `2378120_4_crop_14476.jpg` | excluded | danger | cut: 26.55% | danger: 48.82% | excluded: 24.63% | ❌ |
