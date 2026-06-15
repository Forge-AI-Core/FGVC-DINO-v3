# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `70.00%` (7/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `1976170_0_crop_9832.jpg` | cut | cut | cut: 65.74% | danger: 25.25% | excluded: 9.01% | ✅ |
| 2 | `6_1189931_crop_7140.jpg` | cut | cut | cut: 75.42% | danger: 19.65% | excluded: 4.93% | ✅ |
| 3 | `0_5199660_crop_874.jpg` | cut | cut | cut: 51.21% | danger: 40.81% | excluded: 7.98% | ✅ |
| 4 | `2262963_3_crop_12739.jpg` | excluded | cut | cut: 57.41% | danger: 34.26% | excluded: 8.33% | ❌ |
| 5 | `2193320_0_crop_12542.jpg` | excluded | cut | cut: 58.12% | danger: 28.65% | excluded: 13.23% | ❌ |
| 6 | `Daekwang_2022072100518_crop_15138.jpg` | cut | cut | cut: 58.73% | danger: 36.49% | excluded: 4.78% | ✅ |
| 7 | `808563_crop_7999.jpg` | cut | cut | cut: 51.25% | danger: 40.63% | excluded: 8.12% | ✅ |
| 8 | `2181763_4_crop_12065.jpg` | cut | cut | cut: 70.32% | danger: 22.91% | excluded: 6.76% | ✅ |
| 9 | `Yein_2022070400967_crop_17442.jpg` | cut | cut | cut: 69.75% | danger: 24.67% | excluded: 5.58% | ✅ |
| 10 | `0_5481605_crop_1750.jpg` | danger | cut | cut: 54.53% | danger: 27.61% | excluded: 17.85% | ❌ |
