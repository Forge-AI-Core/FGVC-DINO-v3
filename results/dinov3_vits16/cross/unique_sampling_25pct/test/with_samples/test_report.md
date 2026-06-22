# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `60.00%` (6/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00387_FP_review_real_fp.jpg` | excluded | danger | cut: 3.06% | danger: 96.19% | excluded: 0.75% | ❌ |
| 2 | `00052_original_GT_duplicate_obj59.jpg` | danger | danger | cut: 2.26% | danger: 93.37% | excluded: 4.37% | ✅ |
| 3 | `00247_original_GT_obj252.jpg` | danger | danger | cut: 11.13% | danger: 88.28% | excluded: 0.59% | ✅ |
| 4 | `00014_original_GT_obj8.jpg` | danger | danger | cut: 14.14% | danger: 70.76% | excluded: 15.10% | ✅ |
| 5 | `00262_original_GT_duplicate_obj265.jpg` | danger | cut | cut: 49.98% | danger: 49.18% | excluded: 0.84% | ❌ |
| 6 | `00083_original_GT_duplicate_obj77.jpg` | danger | excluded | cut: 14.51% | danger: 5.78% | excluded: 79.71% | ❌ |
| 7 | `00200_original_GT_obj189.jpg` | danger | cut | cut: 55.75% | danger: 41.72% | excluded: 2.53% | ❌ |
| 8 | `00239_original_GT_obj247.jpg` | danger | danger | cut: 37.01% | danger: 60.84% | excluded: 2.16% | ✅ |
| 9 | `00004_original_GT_duplicate_obj5.jpg` | danger | danger | cut: 5.79% | danger: 93.91% | excluded: 0.31% | ✅ |
| 10 | `00178_original_GT_obj170.jpg` | danger | danger | cut: 2.72% | danger: 97.04% | excluded: 0.23% | ✅ |
