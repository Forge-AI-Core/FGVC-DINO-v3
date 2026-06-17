# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_50pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_50pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `70.00%` (7/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00367_FP_review_drum.jpg` | cut | cut | cut: 35.91% | danger: 33.06% | excluded: 31.03% | ✅ |
| 2 | `00119_original_GT_obj121.jpg` | danger | danger | cut: 22.44% | danger: 74.01% | excluded: 3.55% | ✅ |
| 3 | `00234_original_GT_obj243.jpg` | danger | danger | cut: 11.86% | danger: 84.02% | excluded: 4.12% | ✅ |
| 4 | `00277_hidden_GT_from_FP_review_obj90004.jpg` | danger | danger | cut: 26.49% | danger: 60.32% | excluded: 13.18% | ✅ |
| 5 | `00099_original_GT_obj103.jpg` | danger | danger | cut: 10.03% | danger: 83.85% | excluded: 6.12% | ✅ |
| 6 | `00077_original_GT_duplicate_obj74.jpg` | danger | cut | cut: 55.00% | danger: 41.62% | excluded: 3.38% | ❌ |
| 7 | `00027_original_GT_obj25.jpg` | danger | danger | cut: 35.18% | danger: 54.22% | excluded: 10.60% | ✅ |
| 8 | `00408_FP_review_real_fp.jpg` | excluded | danger | cut: 14.79% | danger: 56.63% | excluded: 28.58% | ❌ |
| 9 | `00252_original_GT_obj256.jpg` | danger | cut | cut: 88.85% | danger: 8.45% | excluded: 2.70% | ❌ |
| 10 | `00350_xlsx_excluded.jpg` | cut | cut | cut: 68.97% | danger: 26.73% | excluded: 4.29% | ✅ |
