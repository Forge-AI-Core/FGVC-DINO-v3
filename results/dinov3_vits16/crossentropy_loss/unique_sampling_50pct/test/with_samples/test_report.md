# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_50pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_50pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `40.00%` (4/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00210_original_GT_obj212.jpg` | danger | danger | cut: 20.15% | danger: 79.74% | excluded: 0.11% | ✅ |
| 2 | `00257_original_GT_obj263.jpg` | danger | danger | cut: 6.40% | danger: 91.78% | excluded: 1.82% | ✅ |
| 3 | `00192_original_GT_obj183.jpg` | danger | cut | cut: 77.37% | danger: 22.49% | excluded: 0.14% | ❌ |
| 4 | `00327_xlsx_excluded.jpg` | excluded | cut | cut: 55.78% | danger: 43.30% | excluded: 0.92% | ❌ |
| 5 | `00177_original_GT_duplicate_obj169.jpg` | danger | danger | cut: 13.31% | danger: 64.90% | excluded: 21.79% | ✅ |
| 6 | `00083_original_GT_duplicate_obj77.jpg` | danger | excluded | cut: 25.85% | danger: 31.31% | excluded: 42.84% | ❌ |
| 7 | `00109_original_GT_duplicate_obj109.jpg` | danger | danger | cut: 2.97% | danger: 94.76% | excluded: 2.27% | ✅ |
| 8 | `00087_original_GT_obj79.jpg` | danger | excluded | cut: 25.48% | danger: 8.81% | excluded: 65.71% | ❌ |
| 9 | `00114_original_GT_duplicate_obj113.jpg` | danger | cut | cut: 87.14% | danger: 12.65% | excluded: 0.21% | ❌ |
| 10 | `00248_original_GT_obj253.jpg` | danger | cut | cut: 73.66% | danger: 25.78% | excluded: 0.56% | ❌ |
