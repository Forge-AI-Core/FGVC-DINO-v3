# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `40.00%` (4/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00117_original_GT_obj116.jpg` | danger | danger | cut: 22.03% | danger: 77.58% | excluded: 0.39% | ✅ |
| 2 | `00109_original_GT_duplicate_obj109.jpg` | danger | danger | cut: 1.02% | danger: 92.94% | excluded: 6.04% | ✅ |
| 3 | `00304_xlsx_excluded.jpg` | cut | danger | cut: 2.24% | danger: 79.95% | excluded: 17.82% | ❌ |
| 4 | `00146_original_GT_duplicate_obj146.jpg` | danger | cut | cut: 96.92% | danger: 2.85% | excluded: 0.22% | ❌ |
| 5 | `00183_original_GT_obj178.jpg` | danger | danger | cut: 1.46% | danger: 98.43% | excluded: 0.11% | ✅ |
| 6 | `00327_xlsx_excluded.jpg` | excluded | cut | cut: 75.30% | danger: 24.39% | excluded: 0.31% | ❌ |
| 7 | `00282_hidden_GT_from_FP_review_obj90009.jpg` | danger | cut | cut: 72.54% | danger: 23.24% | excluded: 4.22% | ❌ |
| 8 | `00240_original_GT_duplicate_obj247.jpg` | danger | cut | cut: 73.50% | danger: 14.33% | excluded: 12.18% | ❌ |
| 9 | `00129_original_GT_duplicate_obj130.jpg` | danger | danger | cut: 7.73% | danger: 92.15% | excluded: 0.12% | ✅ |
| 10 | `00127_original_GT_obj129.jpg` | danger | cut | cut: 45.77% | danger: 25.78% | excluded: 28.44% | ❌ |
