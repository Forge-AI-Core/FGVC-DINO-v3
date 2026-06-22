# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vits16`
- **Dataset Split**: `vanilla_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_vanilla_0pct_dinov3_vits16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `80.00%` (8/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00043_original_GT_obj50.jpg` | danger | danger | cut: 0.86% | danger: 98.88% | excluded: 0.26% | ✅ |
| 2 | `00380_FP_review_drum.jpg` | cut | danger | cut: 40.87% | danger: 57.76% | excluded: 1.37% | ❌ |
| 3 | `00031_original_GT_obj30.jpg` | danger | danger | cut: 1.70% | danger: 97.98% | excluded: 0.32% | ✅ |
| 4 | `00003_original_GT_obj5.jpg` | danger | danger | cut: 8.05% | danger: 91.80% | excluded: 0.15% | ✅ |
| 5 | `00107_original_GT_duplicate_obj109.jpg` | danger | danger | cut: 20.39% | danger: 79.50% | excluded: 0.11% | ✅ |
| 6 | `00234_original_GT_obj243.jpg` | danger | danger | cut: 0.09% | danger: 99.81% | excluded: 0.09% | ✅ |
| 7 | `00263_original_GT_obj266.jpg` | danger | cut | cut: 53.89% | danger: 1.15% | excluded: 44.96% | ❌ |
| 8 | `00077_original_GT_duplicate_obj74.jpg` | danger | danger | cut: 22.83% | danger: 77.13% | excluded: 0.04% | ✅ |
| 9 | `00117_original_GT_obj116.jpg` | danger | danger | cut: 13.25% | danger: 86.59% | excluded: 0.15% | ✅ |
| 10 | `00035_original_GT_obj34.jpg` | danger | danger | cut: 3.75% | danger: 95.55% | excluded: 0.69% | ✅ |
