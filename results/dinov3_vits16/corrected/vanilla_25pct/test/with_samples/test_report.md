# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `60.00%` (6/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `1_5204904_crop_3562.jpg` | excluded | excluded | cut: 32.10% | danger: 10.14% | excluded: 57.76% | ✅ |
| 2 | `2181894_0_crop_12418.jpg` | cut | cut | cut: 99.46% | danger: 0.06% | excluded: 0.48% | ✅ |
| 3 | `0_5203504_crop_960.jpg` | excluded | excluded | cut: 1.17% | danger: 0.48% | excluded: 98.35% | ✅ |
| 4 | `2304552_1_crop_13315.jpg` | cut | cut | cut: 54.30% | danger: 44.43% | excluded: 1.27% | ✅ |
| 5 | `Jaei_2022041501904_crop_16061.jpg` | cut | cut | cut: 96.18% | danger: 3.47% | excluded: 0.35% | ✅ |
| 6 | `Yein_2022091401971_crop_17490.jpg` | excluded | danger | cut: 5.93% | danger: 93.93% | excluded: 0.13% | ❌ |
| 7 | `2323865_2_crop_13694.jpg` | excluded | excluded | cut: 8.78% | danger: 1.92% | excluded: 89.30% | ✅ |
| 8 | `1_5199661_crop_3462.jpg` | excluded | danger | cut: 12.06% | danger: 87.53% | excluded: 0.41% | ❌ |
| 9 | `Yein_2022070400963_crop_17425.jpg` | danger | cut | cut: 91.61% | danger: 8.35% | excluded: 0.04% | ❌ |
| 10 | `2327617_0_crop_13826.jpg` | excluded | cut | cut: 48.88% | danger: 15.20% | excluded: 35.92% | ❌ |
