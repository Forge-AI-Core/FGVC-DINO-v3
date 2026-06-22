# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vitl16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vitl16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `60.00%` (6/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00419_FP_review.jpg` | cut | danger | cut: 7.10% | danger: 92.70% | excluded: 0.20% | ❌ |
| 2 | `00410_FP_review_real_fp.jpg` | excluded | danger | cut: 0.66% | danger: 84.53% | excluded: 14.82% | ❌ |
| 3 | `00231_original_GT_duplicate_obj237.jpg` | danger | cut | cut: 40.33% | danger: 40.18% | excluded: 19.50% | ❌ |
| 4 | `00197_original_GT_duplicate_obj186.jpg` | danger | danger | cut: 0.83% | danger: 98.98% | excluded: 0.19% | ✅ |
| 5 | `00109_original_GT_duplicate_obj109.jpg` | danger | danger | cut: 2.69% | danger: 96.14% | excluded: 1.17% | ✅ |
| 6 | `00104_original_GT_obj106.jpg` | danger | danger | cut: 2.23% | danger: 97.55% | excluded: 0.22% | ✅ |
| 7 | `00006_original_GT_duplicate_obj5.jpg` | danger | danger | cut: 5.02% | danger: 93.58% | excluded: 1.40% | ✅ |
| 8 | `00171_original_GT_obj164.jpg` | danger | danger | cut: 0.89% | danger: 98.74% | excluded: 0.37% | ✅ |
| 9 | `00258_original_GT_duplicate_obj263.jpg` | danger | danger | cut: 44.95% | danger: 53.85% | excluded: 1.21% | ✅ |
| 10 | `00411_FP_review_real_fp.jpg` | excluded | danger | cut: 1.15% | danger: 80.37% | excluded: 18.48% | ❌ |
