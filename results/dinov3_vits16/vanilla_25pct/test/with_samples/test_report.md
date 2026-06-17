# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `20.00%` (2/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00070_original_GT_obj71.jpg` | danger | cut | cut: 65.38% | danger: 22.94% | excluded: 11.68% | ❌ |
| 2 | `00266_original_GT_duplicate_obj267.jpg` | danger | cut | cut: 53.37% | danger: 39.38% | excluded: 7.25% | ❌ |
| 3 | `00126_original_GT_obj128.jpg` | danger | cut | cut: 79.49% | danger: 17.03% | excluded: 3.49% | ❌ |
| 4 | `00168_original_GT_obj162.jpg` | danger | cut | cut: 65.69% | danger: 21.15% | excluded: 13.15% | ❌ |
| 5 | `00066_original_GT_obj69.jpg` | danger | cut | cut: 58.21% | danger: 28.45% | excluded: 13.34% | ❌ |
| 6 | `00256_original_GT_obj262.jpg` | danger | danger | cut: 13.23% | danger: 76.07% | excluded: 10.69% | ✅ |
| 7 | `00411_FP_review_real_fp.jpg` | excluded | danger | cut: 17.38% | danger: 53.96% | excluded: 28.66% | ❌ |
| 8 | `00114_original_GT_duplicate_obj113.jpg` | danger | cut | cut: 89.19% | danger: 7.76% | excluded: 3.05% | ❌ |
| 9 | `00215_original_GT_duplicate_obj214.jpg` | danger | cut | cut: 63.45% | danger: 22.34% | excluded: 14.21% | ❌ |
| 10 | `00258_original_GT_duplicate_obj263.jpg` | danger | danger | cut: 30.42% | danger: 60.96% | excluded: 8.62% | ✅ |
