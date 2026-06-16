from pathlib import Path
from typing import Any
import argparse
import yaml
import torch

from linear_head_focal_loss_supercon.train_test_elements import (
    get_modules,
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

    num_epochs = hyperparams["train"]["NUM_EPOCHS"]
    learning_rate = float(hyperparams["train"]["LEARNING_RATE"])
    weight_decay = float(hyperparams["train"]["WEIGHT_DECAY"])
    early_stopping_patience = hyperparams["train"]["EARLY_STOPPING_PATIENCE"]
    accumulation_steps = hyperparams["train"].get("ACCUMULATION_STEPS", 1)
    stage1_batch_size = hyperparams["train"].get("STAGE1_BATCH_SIZE", batch_size)

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

    checkpoint_dir = Path(hyperparams["test"]["linear_lora"]["CHECKPOINT_DIR"])
    checkpoint_path = checkpoint_dir / f"best_model_{dataset_name}_{model_name}.pth"

    # 모듈 초기화
    (
        train_loader,
        val_loader,
        test_loader,
        device,
        model,
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
    )

    # patch_size 유추 (dino 모델 특성)
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    patch_size = base_vit.patch_size if hasattr(base_vit, "patch_size") else 14

    # 파이프라인 실행 (조립)
    
    # 총 에폭을 50:50 비율로 분할
    num_epochs_stage1 = int(num_epochs * 0.5)
    num_epochs_stage2 = num_epochs - num_epochs_stage1

    # Stage 1: Representation Learning (SupCon)
    from linear_head_focal_loss_supercon.train_test_elements import (
        run_stage1_supcon,
        run_stage2_focal,
        freeze_for_stage2,
    )
    
    checkpoint_stage1 = checkpoint_dir / f"stage1_best_model_{dataset_name}_{model_name}.pth"
    run_stage1_supcon(
        num_epochs=num_epochs_stage1,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        accumulation_steps=accumulation_steps,
        stage1_batch_size=stage1_batch_size,
        checkpoint_path=checkpoint_stage1,
    )
    
    # Load Best Stage 1 Model
    if checkpoint_stage1.exists():
        model.load_state_dict(torch.load(checkpoint_stage1, weights_only=True))
        
    # Freeze for Stage 2
    freeze_for_stage2(model)
    
    # Stage 2: Classifier Fine-Tuning (Focal Loss)
    run_stage2_focal(
        num_epochs=num_epochs_stage2,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
