# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `80.00%` (8/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `1960100_1_crop_8891.jpg` | cut | cut | cut: 50.51% | danger: 32.46% | excluded: 17.03% | ✅ |
| 2 | `Dado_2022031101464_crop_14960.jpg` | cut | cut | cut: 58.75% | danger: 36.32% | excluded: 4.94% | ✅ |
| 3 | `1981745_2_crop_10235.jpg` | excluded | cut | cut: 56.32% | danger: 20.05% | excluded: 23.63% | ❌ |
| 4 | `1980011_0_crop_10130.jpg` | danger | danger | cut: 8.46% | danger: 83.46% | excluded: 8.09% | ✅ |
| 5 | `2048321_0_crop_11150.jpg` | excluded | excluded | cut: 11.29% | danger: 15.44% | excluded: 73.28% | ✅ |
| 6 | `1960224_0_crop_9162.jpg` | cut | cut | cut: 72.63% | danger: 20.09% | excluded: 7.28% | ✅ |
| 7 | `0_5505745_crop_2227.jpg` | danger | cut | cut: 47.40% | danger: 43.68% | excluded: 8.92% | ❌ |
| 8 | `2355042_4_crop_14219.jpg` | excluded | excluded | cut: 26.27% | danger: 28.87% | excluded: 44.87% | ✅ |
| 9 | `2140019_0_crop_11732.jpg` | danger | danger | cut: 32.66% | danger: 60.23% | excluded: 7.11% | ✅ |
| 10 | `2181764_5_crop_12087.jpg` | cut | cut | cut: 75.66% | danger: 18.86% | excluded: 5.48% | ✅ |
