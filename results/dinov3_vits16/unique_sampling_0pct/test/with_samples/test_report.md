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
| 1 | `00068_original_GT_duplicate_obj70.jpg` | danger | cut | cut: 58.56% | danger: 41.15% | excluded: 0.29% | ❌ |
| 2 | `00189_original_GT_obj181.jpg` | danger | danger | cut: 5.37% | danger: 94.52% | excluded: 0.11% | ✅ |
| 3 | `00246_original_GT_obj251.jpg` | danger | danger | cut: 7.62% | danger: 92.29% | excluded: 0.09% | ✅ |
| 4 | `00305_xlsx_excluded.jpg` | cut | cut | cut: 61.76% | danger: 5.76% | excluded: 32.48% | ✅ |
| 5 | `00294_hidden_GT_from_FP_review_obj90021.jpg` | danger | danger | cut: 11.10% | danger: 88.58% | excluded: 0.32% | ✅ |
| 6 | `00289_hidden_GT_from_FP_review_obj90016.jpg` | danger | danger | cut: 3.25% | danger: 96.66% | excluded: 0.09% | ✅ |
| 7 | `00000_original_GT_obj2.jpg` | danger | danger | cut: 12.03% | danger: 79.86% | excluded: 8.11% | ✅ |
| 8 | `00267_original_GT_obj268.jpg` | danger | cut | cut: 86.31% | danger: 9.93% | excluded: 3.76% | ❌ |
| 9 | `00158_original_GT_obj151.jpg` | danger | danger | cut: 7.78% | danger: 89.70% | excluded: 2.53% | ✅ |
| 10 | `00177_original_GT_duplicate_obj169.jpg` | danger | danger | cut: 12.76% | danger: 85.71% | excluded: 1.53% | ✅ |
