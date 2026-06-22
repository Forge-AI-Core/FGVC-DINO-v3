from pathlib import Path
from typing import Any
import argparse
import yaml

from linear_head_no_lora.train_test_elements import (
    get_modules,
    run_train_val_process,
    run_testset_process,
    run_sample_test_process,
)


def load_hyperparams(file_path: Path) -> dict[str, Any]:
    """하이퍼파라미터 로드 함수"""
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)

    return hyperparams


def parse_args() -> argparse.Namespace:
    """CLI 인자 파싱 함수"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hyperparams",
        type=str,
        default="./hyperparams.yaml",
        required=False,
        help="Path to hyperparameter file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=False,
        help="Model name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        required=False,
        help="Dataset name.",
    )

    return parser.parse_args()


def main():
    """메인 실행 함수"""
    # cli 인자 파싱
    args = parse_args()

    # 하이퍼 파라미터 로드
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(file_path=hyperparam_path)

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

    # CLI args로 넘어오지 않았을 경우 input
    model_name = (
        input(f"Enter model name ({', '.join(model_options)}): ").lower().strip()
    )
    dataset_name = (
        input(f"Enter dataset name ({', '.join(dataset_options)}): ").lower().strip()
    )

    model_config = hyperparams["model"][model_name]
    model_path = Path(model_config["PATH"])
    model_name = model_config["NAME"]

    dataset_config = hyperparams["data"][dataset_name]
    dataset_dir = Path(dataset_config["DATASET_DIR"])
    dataset_name = dataset_config["DATASET_NAME"]

    checkpoint_dir = Path(hyperparams["test"]["linear_lora"]["CHECKPOINT_DIR"]) / "no_lora"
    checkpoint_path = checkpoint_dir / f"best_model_{dataset_name}_{model_name}.pth"

    # 모듈 초기화
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

    # patch_size 유추 (dino 모델 특성)
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    patch_size = base_vit.patch_size if hasattr(base_vit, "patch_size") else 14

    # 파이프라인 실행 (조립)
    run_train_val_process(
        num_epochs=num_epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        dataset_name=dataset_name,
        model_name=model_name,
        hyperparam_path=hyperparam_path,
    )

    run_testset_process(
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        image_size=image_size,
        test_loader=test_loader,
        hyperparam_path=hyperparam_path,
    )

    run_sample_test_process(
        model_name=model_name,
        test_loader=test_loader,
        dataset_name=dataset_name,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        patch_size=patch_size,
    )


if __name__ == "__main__":
    main()
