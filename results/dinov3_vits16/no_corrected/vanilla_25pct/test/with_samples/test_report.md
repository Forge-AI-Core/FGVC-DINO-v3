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
| 1 | `4_5200974_crop_6158.jpg` | excluded | cut | cut: 95.47% | danger: 0.32% | excluded: 4.21% | ❌ |
| 2 | `6_5201534_crop_7194.jpg` | cut | cut | cut: 84.56% | danger: 13.65% | excluded: 1.79% | ✅ |
| 3 | `4_5200416_crop_6140.jpg` | excluded | cut | cut: 51.96% | danger: 18.45% | excluded: 29.59% | ❌ |
| 4 | `Hyunil_2022072001677_crop_15743.jpg` | cut | cut | cut: 65.21% | danger: 34.59% | excluded: 0.21% | ✅ |
| 5 | `5_1229398_crop_6676.jpg` | danger | cut | cut: 52.23% | danger: 47.07% | excluded: 0.71% | ❌ |
| 6 | `4_5202309_crop_6209.jpg` | cut | cut | cut: 86.91% | danger: 8.81% | excluded: 4.28% | ✅ |
| 7 | `9_0_1495479_original_crop_7637.jpg` | danger | cut | cut: 48.40% | danger: 30.93% | excluded: 20.68% | ❌ |
| 8 | `3_5472459_crop_5654.jpg` | cut | cut | cut: 96.39% | danger: 0.63% | excluded: 2.98% | ✅ |
| 9 | `2_0_1409656_original_crop_3777.jpg` | danger | cut | cut: 50.58% | danger: 42.83% | excluded: 6.59% | ❌ |
| 10 | `4_1182182_crop_6005.jpg` | danger | danger | cut: 37.09% | danger: 39.73% | excluded: 23.18% | ✅ |
