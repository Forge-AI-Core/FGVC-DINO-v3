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
| 1 | `1_2_1036762_crop_2762.jpg` | cut | cut | cut: 99.59% | danger: 0.27% | excluded: 0.14% | ✅ |
| 2 | `2_5_1409661_original_crop_4135.jpg` | danger | danger | cut: 0.59% | danger: 99.33% | excluded: 0.08% | ✅ |
| 3 | `1_5204244_crop_3552.jpg` | excluded | excluded | cut: 47.57% | danger: 3.90% | excluded: 48.53% | ✅ |
| 4 | `1964109_0_crop_9442.jpg` | cut | cut | cut: 97.90% | danger: 0.56% | excluded: 1.54% | ✅ |
| 5 | `Bookwang_2022030500909_crop_14574.jpg` | danger | cut | cut: 92.69% | danger: 6.98% | excluded: 0.33% | ❌ |
| 6 | `1999847_2_crop_10413.jpg` | cut | cut | cut: 75.54% | danger: 24.01% | excluded: 0.45% | ✅ |
| 7 | `1959946_1_crop_8725.jpg` | cut | cut | cut: 98.90% | danger: 0.19% | excluded: 0.91% | ✅ |
| 8 | `2002530_0_crop_10480.jpg` | cut | danger | cut: 9.97% | danger: 89.93% | excluded: 0.10% | ❌ |
| 9 | `Dongbo_2022052001111_crop_15234.jpg` | cut | cut | cut: 99.24% | danger: 0.68% | excluded: 0.08% | ✅ |
| 10 | `2_1_1036783_crop_3902.jpg` | danger | excluded | cut: 20.46% | danger: 27.32% | excluded: 52.23% | ❌ |
