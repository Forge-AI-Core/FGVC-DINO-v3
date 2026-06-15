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
| 1 | `1959961_0_crop_8729.jpg` | danger | cut | cut: 62.74% | danger: 26.65% | excluded: 10.60% | ❌ |
| 2 | `1_2_1425135_original_crop_2871.jpg` | danger | cut | cut: 71.78% | danger: 21.21% | excluded: 7.02% | ❌ |
| 3 | `0_1182178_crop_185.jpg` | cut | cut | cut: 65.90% | danger: 24.40% | excluded: 9.70% | ✅ |
| 4 | `0_5470750_crop_1552.jpg` | danger | cut | cut: 45.68% | danger: 32.59% | excluded: 21.74% | ❌ |
| 5 | `2325046_1_crop_13735.jpg` | danger | cut | cut: 55.60% | danger: 32.66% | excluded: 11.75% | ❌ |
| 6 | `2017550_0_crop_10676.jpg` | excluded | cut | cut: 54.45% | danger: 30.03% | excluded: 15.52% | ❌ |
| 7 | `2181764_5_crop_12071.jpg` | cut | cut | cut: 76.51% | danger: 17.23% | excluded: 6.26% | ✅ |
| 8 | `3_5465664_crop_5591.jpg` | cut | cut | cut: 69.45% | danger: 22.46% | excluded: 8.10% | ✅ |
| 9 | `3_5488843_crop_5683.jpg` | cut | cut | cut: 60.31% | danger: 27.72% | excluded: 11.97% | ✅ |
| 10 | `2309992_0_crop_13464.jpg` | excluded | cut | cut: 52.94% | danger: 32.32% | excluded: 14.74% | ❌ |
