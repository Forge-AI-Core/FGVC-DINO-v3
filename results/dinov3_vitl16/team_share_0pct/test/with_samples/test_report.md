# DINOv3 Inference Evaluation Report

## 📋 Run Metadata
- **Model Backbone**: `dinov3_vitl16`
- **Dataset Split**: `team_share_0pct`
- **Checkpoint Path**: `models/dino/weights/linear-peft-lora/best_model_team_share_0pct_dinov3_vitl16.pth`
- **Total Samples Evaluated**: `10`
- **Overall Accuracy**: `100.00%` (10/10)

## 🔍 Detailed Inference Results

| Index | File Name | True Class | Predicted Class | Probabilities | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| 1 | `00224_original_GT_obj231.jpg` | danger | danger | cut: 0.30% | danger: 98.76% | excluded: 0.95% | ✅ |
| 2 | `00001_original_GT_duplicate_obj2.jpg` | danger | danger | cut: 0.54% | danger: 99.42% | excluded: 0.04% | ✅ |
| 3 | `00239_original_GT_obj247.jpg` | danger | danger | cut: 4.87% | danger: 94.91% | excluded: 0.22% | ✅ |
| 4 | `00161_original_GT_duplicate_obj151.jpg` | danger | danger | cut: 7.04% | danger: 74.40% | excluded: 18.56% | ✅ |
| 5 | `00237_original_GT_duplicate_obj246.jpg` | danger | danger | cut: 34.93% | danger: 64.89% | excluded: 0.18% | ✅ |
| 6 | `00184_original_GT_duplicate_obj178.jpg` | danger | danger | cut: 20.78% | danger: 78.28% | excluded: 0.95% | ✅ |
| 7 | `00280_hidden_GT_from_FP_review_obj90007.jpg` | danger | danger | cut: 0.56% | danger: 99.15% | excluded: 0.28% | ✅ |
| 8 | `00284_hidden_GT_from_FP_review_obj90011.jpg` | danger | danger | cut: 32.14% | danger: 42.34% | excluded: 25.53% | ✅ |
| 9 | `00373_FP_review_drum.jpg` | cut | cut | cut: 85.94% | danger: 11.16% | excluded: 2.90% | ✅ |
| 10 | `00155_original_GT_duplicate_obj147.jpg` | danger | danger | cut: 2.01% | danger: 97.72% | excluded: 0.27% | ✅ |
