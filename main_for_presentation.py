import argparse
import yaml
from pathlib import Path
from PIL import Image

import torch
from torchvision import transforms

from linear_head.train_test_elements import get_modules
from linear_head.get_data_loaders import SquarePad
from linear_head.test_utils.test_metrics_utils_for_samples import (
    patch_attention_module,
    generate_attention_heatmap,
    generate_gradcam_heatmap,
    save_visualization_plot,
)
import linear_head.test_utils.test_metrics_utils_for_samples as tmus


def load_hyperparams(file_path: Path) -> dict:
    """하이퍼파라미터 로드 함수"""
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """CLI 인자 파싱 함수"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, default="./hyperparams.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(hyperparam_path)

    batch_size = hyperparams["data"]["BATCH_SIZE"]
    image_size = hyperparams["data"]["IMAGE_SIZE"]

    num_classes = hyperparams["model"]["NUM_CLASSES"]
    hidden_dim1 = hyperparams["model"]["HIDDEN_DIM1"]
    hidden_dim2 = hyperparams["model"]["HIDDEN_DIM2"]
    lora_rank = hyperparams["model"]["LORA_RANK"]
    lora_alpha = hyperparams["model"]["LORA_ALPHA"]
    target_modules = hyperparams["model"]["TARGET_MODULES"]
    lora_dropout = hyperparams["model"].get("LORA_DROPOUT", 0.5)
    head_dropout = hyperparams["model"].get("HEAD_DROPOUT", 0.5)
    label_smoothing = hyperparams["model"].get("LABEL_SMOOTHING", 0.0)

    num_epochs = hyperparams["train"]["NUM_EPOCHS"]
    learning_rate = float(hyperparams["train"]["LEARNING_RATE"])
    weight_decay = float(hyperparams["train"]["WEIGHT_DECAY"])
    early_stopping_patience = hyperparams["train"]["EARLY_STOPPING_PATIENCE"]
    accumulation_steps = hyperparams["train"].get("ACCUMULATION_STEPS", 1)

    # 설정 파일로부터 지원 가능한 모델 및 데이터셋 목록을 동적으로 파싱
    model_options = [
        k
        for k, v in hyperparams.get("model", {}).items()
        if isinstance(v, dict) and "PATH" in v
    ]
    dataset_options = [
        k
        for k, v in hyperparams.get("data", {}).items()
        if isinstance(v, dict) and "DATASET_DIR" in v
    ]

    model_name_key = (
        input(f"Enter model name ({', '.join(model_options)}): ").lower().strip()
    )
    dataset_name_key = (
        input(f"Enter dataset name ({', '.join(dataset_options)}): ").lower().strip()
    )

    model_config = hyperparams["model"][model_name_key]
    model_path = Path(model_config["PATH"])
    model_name = model_config["NAME"]

    dataset_config = hyperparams["data"][dataset_name_key]
    dataset_dir = Path(dataset_config["DATASET_DIR"])
    dataset_name = dataset_config["DATASET_NAME"]

    checkpoint_dir = Path(hyperparams["test"]["linear_lora"]["CHECKPOINT_DIR"])
    checkpoint_path = checkpoint_dir / f"best_model_{dataset_name}_{model_name}.pth"

    # get_modules를 활용해 모델 로드 (데이터로더 등은 셋업용)
    (
        train_loader,
        val_loader,
        test_loader,
        device,
        model,
        criterion,
        optimizer,
        lr_scheduler,
    ) = get_modules(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        model_name=model_name,
        model_path=model_path,
        num_classes=num_classes,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        accumulation_steps=accumulation_steps,
        head_dropout=head_dropout,
        lora_dropout=lora_dropout,
        label_smoothing=label_smoothing,
    )

    # 체크포인트 로드 (Grad-CAM을 위해 학습된 가중치 필수)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    # 어텐션 몽키 패치 적용
    patch_attention_module(model)

    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    patch_size = base_vit.patch_size if hasattr(base_vit, "patch_size") else 14

    samples_dir = Path("./data/Iron-Scraps/samples_for_presentation")
    results_dir = Path(f"./results/presentation_heatmaps/{model_name}_{dataset_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 검증용 트랜스폼 설정 (test 데이터셋 처리와 동일)
    val_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class_names = test_loader.dataset.classes
    print(f"[*] Loaded model '{model_name}' trained on '{dataset_name}'.")
    print(f"[*] Processing images in {samples_dir}...")

    image_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))

    for idx, img_path in enumerate(image_files):
        raw_img = Image.open(img_path).convert("RGB")
        img_tensor = val_transform(raw_img)
        img_input = img_tensor.unsqueeze(0).to(device)
        img_input.requires_grad_(True)

        # Grad-CAM을 위해 inference_mode 해제 및 gradients 계산
        model.zero_grad()
        logits = model(img_input)
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

        pred_label = int(probs.argmax())
        confidence = float(probs[pred_label])
        pred_name = class_names[pred_label]

        # 분류기가 결정한 클래스(pred_label)에 대한 역전파 수행
        target_logit = logits[0, pred_label]
        target_logit.backward(retain_graph=True)

        # 파일명에서 힌트가 있지만, 폴더 내 임의의 이미지라고 가정하여 Unknown 처리 (필요시 수정)
        true_name = "Unknown"

        heatmap = generate_attention_heatmap(
            model=model,
            img_tensor=img_input,
            image_size=image_size,
            patch_size=patch_size,
        )

        gradcam_heatmap = generate_gradcam_heatmap(
            model=model,
            img_tensor=img_input,
            image_size=image_size,
            patch_size=patch_size,
        )

        # 다음 루프를 위해 캐시된 어텐션 및 그래디언트 초기화
        tmus.captured_attn_weights = None
        tmus.captured_attn_gradients = None

        orig_img = SquarePad()(raw_img)
        save_path = results_dir / f"attention_gradcam_{img_path.stem}.png"

        save_visualization_plot(
            orig_img=orig_img,
            heatmap=heatmap,
            gradcam_heatmap=gradcam_heatmap,
            true_label_name=true_name,
            pred_label_name=pred_name,
            confidence=confidence,
            save_path=save_path,
        )
        print(
            f"[{idx+1}/{len(image_files)}] Saved: {save_path.name} (Pred: {pred_name} | {confidence*100:.1f}%)"
        )

    print(f"[*] Finished processing all {len(image_files)} images.")
    print(f"[*] Heatmaps are saved in: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
