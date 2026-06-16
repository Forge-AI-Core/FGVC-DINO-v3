# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `50.00%` (5/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00174_original_GT_obj166.jpg` | danger | danger | cut: 30.10% | danger: 66.94% | excluded: 2.96% | ✅ |
| 2 | `00401_FP_review_real_fp.jpg` | cut | danger | cut: 47.49% | danger: 49.96% | excluded: 2.55% | ❌ |
| 3 | `00173_original_GT_duplicate_obj164.jpg` | danger | cut | cut: 55.12% | danger: 44.47% | excluded: 0.41% | ❌ |
| 4 | `00410_FP_review_real_fp.jpg` | excluded | danger | cut: 32.97% | danger: 66.31% | excluded: 0.73% | ❌ |
| 5 | `00285_hidden_GT_from_FP_review_obj90012.jpg` | danger | danger | cut: 47.99% | danger: 51.57% | excluded: 0.44% | ✅ |
| 6 | `00172_original_GT_duplicate_obj164.jpg` | danger | cut | cut: 54.00% | danger: 45.65% | excluded: 0.35% | ❌ |
| 7 | `00158_original_GT_obj151.jpg` | danger | danger | cut: 25.59% | danger: 72.68% | excluded: 1.73% | ✅ |
| 8 | `00309_xlsx_excluded.jpg` | excluded | excluded | cut: 10.78% | danger: 15.72% | excluded: 73.50% | ✅ |
| 9 | `00074_original_GT_obj73.jpg` | danger | cut | cut: 50.22% | danger: 49.36% | excluded: 0.41% | ❌ |
| 10 | `00239_original_GT_obj247.jpg` | danger | danger | cut: 31.78% | danger: 65.77% | excluded: 2.45% | ✅ |
