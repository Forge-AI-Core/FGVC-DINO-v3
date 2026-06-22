# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vitl16`
- **Dataset Split**: `unique_sampling_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_0pct_dinov3_vitl16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `90.00%` (9/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00130_original_GT_duplicate_obj130.jpg` | danger | danger | cut: 2.04% | danger: 97.80% | excluded: 0.16% | ✅ |
| 2 | `00386_FP_review_real_fp.jpg` | excluded | excluded | cut: 2.57% | danger: 39.94% | excluded: 57.49% | ✅ |
| 3 | `00158_original_GT_obj151.jpg` | danger | danger | cut: 2.35% | danger: 97.43% | excluded: 0.23% | ✅ |
| 4 | `00112_original_GT_duplicate_obj112.jpg` | danger | danger | cut: 8.19% | danger: 90.00% | excluded: 1.82% | ✅ |
| 5 | `00134_original_GT_obj133.jpg` | danger | danger | cut: 7.35% | danger: 92.57% | excluded: 0.08% | ✅ |
| 6 | `00012_original_GT_duplicate_obj6.jpg` | danger | cut | cut: 67.54% | danger: 31.98% | excluded: 0.48% | ❌ |
| 7 | `00256_original_GT_obj262.jpg` | danger | danger | cut: 1.29% | danger: 98.15% | excluded: 0.56% | ✅ |
| 8 | `00018_original_GT_obj14.jpg` | danger | danger | cut: 2.96% | danger: 96.26% | excluded: 0.78% | ✅ |
| 9 | `00060_original_GT_obj66.jpg` | danger | danger | cut: 2.61% | danger: 96.96% | excluded: 0.43% | ✅ |
| 10 | `00057_original_GT_obj61.jpg` | danger | danger | cut: 3.67% | danger: 96.21% | excluded: 0.12% | ✅ |
