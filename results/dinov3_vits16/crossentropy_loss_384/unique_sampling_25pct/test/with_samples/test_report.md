# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `unique_sampling_25pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_unique_sampling_25pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `80.00%` (8/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00256_original_GT_obj262.jpg` | danger | danger | cut: 0.41% | danger: 99.36% | excluded: 0.23% | ✅ |
| 2 | `00179_original_GT_obj171.jpg` | danger | danger | cut: 2.46% | danger: 95.04% | excluded: 2.50% | ✅ |
| 3 | `00079_original_GT_duplicate_obj74.jpg` | danger | cut | cut: 65.74% | danger: 33.98% | excluded: 0.28% | ❌ |
| 4 | `00016_original_GT_obj9.jpg` | danger | danger | cut: 0.24% | danger: 99.45% | excluded: 0.31% | ✅ |
| 5 | `00233_original_GT_obj240.jpg` | danger | danger | cut: 38.93% | danger: 60.90% | excluded: 0.17% | ✅ |
| 6 | `00246_original_GT_obj251.jpg` | danger | danger | cut: 1.51% | danger: 98.42% | excluded: 0.07% | ✅ |
| 7 | `00067_original_GT_obj70.jpg` | danger | danger | cut: 0.69% | danger: 99.14% | excluded: 0.17% | ✅ |
| 8 | `00087_original_GT_obj79.jpg` | danger | cut | cut: 69.91% | danger: 8.84% | excluded: 21.25% | ❌ |
| 9 | `00036_original_GT_obj35.jpg` | cut | cut | cut: 57.05% | danger: 42.04% | excluded: 0.91% | ✅ |
| 10 | `00125_original_GT_obj123.jpg` | danger | danger | cut: 18.96% | danger: 42.88% | excluded: 38.16% | ✅ |
