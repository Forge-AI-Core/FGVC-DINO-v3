# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `80.00%` (8/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00028_original_GT_obj26.jpg` | danger | excluded | cut: 13.43% | danger: 28.52% | excluded: 58.06% | ❌ |
| 2 | `00123_original_GT_duplicate_obj122.jpg` | danger | danger | cut: 5.67% | danger: 91.44% | excluded: 2.89% | ✅ |
| 3 | `00419_FP_review.jpg` | cut | cut | cut: 93.31% | danger: 2.72% | excluded: 3.96% | ✅ |
| 4 | `00107_original_GT_duplicate_obj109.jpg` | danger | danger | cut: 8.52% | danger: 91.31% | excluded: 0.17% | ✅ |
| 5 | `00124_original_GT_duplicate_obj122.jpg` | danger | danger | cut: 6.19% | danger: 93.45% | excluded: 0.37% | ✅ |
| 6 | `00338_xlsx_excluded.jpg` | cut | cut | cut: 79.73% | danger: 19.83% | excluded: 0.44% | ✅ |
| 7 | `00098_original_GT_obj98.jpg` | danger | danger | cut: 12.71% | danger: 78.96% | excluded: 8.34% | ✅ |
| 8 | `00224_original_GT_obj231.jpg` | danger | danger | cut: 2.87% | danger: 93.08% | excluded: 4.05% | ✅ |
| 9 | `00416_FP_review_real_fp.jpg` | cut | cut | cut: 67.21% | danger: 31.59% | excluded: 1.20% | ✅ |
| 10 | `00229_original_GT_obj236.jpg` | danger | cut | cut: 67.01% | danger: 5.33% | excluded: 27.66% | ❌ |
