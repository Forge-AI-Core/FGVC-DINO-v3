"""Optuna HPO for Single-Stage Linear Head pipeline.

핵심 탐색 파라미터:
- learning_rate, weight_decay
- lora_rank, lora_alpha, lora_dropout
- hidden_dim1, hidden_dim2, dropout_rate
- label_smoothing
해당 폴더(linear_head)의 단일 Stage 학습 파이프라인(CrossEntropy)을 그대로 사용합니다.
"""

from pathlib import Path
from typing import Any
import sys

# 프로젝트 루트(FGVC-DINO-v3)를 sys.path에 추가하여 'linear_head' 모듈을 찾을 수 있게 합니다.
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import optuna
import torch

from linear_head_no_lora.main import load_hyperparams
from linear_head_no_lora.train_test_elements import get_modules
from linear_head_no_lora.train import train_model
from linear_head_no_lora.validate import validate_model


# ──────────────────────────────────────────────
# 설정 로드 (모듈 로드 시 input 호출 방지)
# ──────────────────────────────────────────────
HYPERPARAMS_PATH = Path("hyperparams.yaml")


def _resolve_config(
    hyperparams: dict[str, Any],
) -> tuple[str, Path, str, Path]:
    """설정 파일로부터 모델/데이터셋 선택지를 파싱하고 사용자 입력을 받아 config를 반환합니다."""
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

    model_key = (
        input(f"Enter model name ({', '.join(model_options)}): ").lower().strip()
    )
    dataset_key = (
        input(f"Enter dataset name ({', '.join(dataset_options)}): ").lower().strip()
    )

    model_config = hyperparams["model"][model_key]
    dataset_config = hyperparams["data"][dataset_key]

    return (
        model_config["NAME"],
        Path(model_config["PATH"]),
        dataset_config["DATASET_NAME"],
        Path(dataset_config["DATASET_DIR"]),
    )


# ──────────────────────────────────────────────
# 학습 파이프라인
# ──────────────────────────────────────────────
def _run_single_stage_pipeline(
    hyperparams: dict[str, Any],
    model_name: str,
    model_path: Path,
    dataset_dir: Path,
    learning_rate: float,
    weight_decay: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    hidden_dim1: int,
    hidden_dim2: int,
    head_dropout: float,
    label_smoothing: float,
) -> float:
    """단일 Stage 학습을 수행하고 best PR-AUC를 반환합니다."""
    batch_size = hyperparams["data"]["BATCH_SIZE"]
    image_size = hyperparams["data"]["IMAGE_SIZE"]
    num_classes = hyperparams["model"]["NUM_CLASSES"]
    target_modules = hyperparams["model"]["TARGET_MODULES"]

    num_epochs = hyperparams["train"]["NUM_EPOCHS"]
    early_stopping_patience = hyperparams["train"]["EARLY_STOPPING_PATIENCE"]
    accumulation_steps = hyperparams["train"].get("ACCUMULATION_STEPS", 1)

    # 모듈 초기화
    (
        train_loader,
        val_loader,
        _,
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

    best_pr_auc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(
            device=device,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
            accumulation_steps=accumulation_steps,
        )

        (
            val_loss,
            val_acc,
            _,
            _,
            _,
            _,
            pr_auc,
            _,
        ) = validate_model(
            val_loader=val_loader,
            model=model,
            device=device,
            criterion=criterion,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        lr_scheduler.step()

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # VRAM 해제
    del model
    del criterion
    del optimizer
    torch.cuda.empty_cache()

    return best_pr_auc


# ──────────────────────────────────────────────
# Optuna Objective
# ──────────────────────────────────────────────
def _create_objective(
    hyperparams: dict[str, Any],
    model_name: str,
    model_path: Path,
    dataset_dir: Path,
):
    """클로저로 objective 함수를 생성하여 모듈 레벨 전역 변수 의존을 제거합니다."""

    def objective(trial: optuna.Trial) -> float:
        # 1. 모델 헤드 파라미터 탐색
        hidden_dim1: int = trial.suggest_categorical("hidden_dim1", [256, 512, 1024])
        # hidden_dim2는 hidden_dim1 이하가 되도록 설정
        hidden_dim2_candidates = [d for d in [128, 256, 512] if d <= hidden_dim1]
        hidden_dim2: int = trial.suggest_categorical(
            "hidden_dim2", hidden_dim2_candidates
        )
        head_dropout: float = trial.suggest_categorical("head_dropout", [0.1, 0.3, 0.5])

        # 2. LoRA 세부 파라미터 탐색
        lora_rank: int = trial.suggest_categorical("lora_rank", [4, 8, 16])
        lora_alpha: int = lora_rank * 2
        lora_dropout: float = trial.suggest_categorical(
            "lora_dropout", [0.0, 0.05, 0.1, 0.2]
        )

        # 3. 정규화 및 Loss 파라미터 탐색
        label_smoothing: float = trial.suggest_categorical(
            "label_smoothing", [0.0, 0.05, 0.1, 0.15]
        )

        # 4. Optimizer 파라미터 탐색
        learning_rate: float = trial.suggest_float(
            "learning_rate", 1e-5, 3e-4, log=True
        )
        weight_decay: float = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)

        print("\n" + "=" * 50)
        print(f"[Optuna Trial #{trial.number}]")
        print(f"  HIDDEN_DIMS   : {hidden_dim1} -> {hidden_dim2}")
        print(f"  HEAD_DROPOUT  : {head_dropout}")
        print(f"  LORA_RANK     : {lora_rank}")
        print(f"  LORA_ALPHA    : {lora_alpha}")
        print(f"  LORA_DROPOUT  : {lora_dropout}")
        print(f"  LABEL_SMOOTH  : {label_smoothing}")
        print(f"  LEARNING_RATE : {learning_rate:.6f}")
        print(f"  WEIGHT_DECAY  : {weight_decay:.5f}")
        print("=" * 50 + "\n")

        best_pr_auc = _run_single_stage_pipeline(
            hyperparams=hyperparams,
            model_name=model_name,
            model_path=model_path,
            dataset_dir=dataset_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            head_dropout=head_dropout,
            label_smoothing=label_smoothing,
        )

        return best_pr_auc

    return objective


# ──────────────────────────────────────────────
# Study Runner
# ──────────────────────────────────────────────
N_TRIALS = 30
TPE_SEED = 2026


def run_hpo(n_trials: int = N_TRIALS) -> None:
    """하이퍼파라미터 최적화 Study를 생성하고 탐색을 실행합니다."""
    hyperparams = load_hyperparams(file_path=HYPERPARAMS_PATH)
    model_name, model_path, _, dataset_dir = _resolve_config(hyperparams)

    objective = _create_objective(
        hyperparams=hyperparams,
        model_name=model_name,
        model_path=model_path,
        dataset_dir=dataset_dir,
    )

    study = optuna.create_study(
        study_name="dinov3_linear_head_hpo",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
    )

    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 50)
    print("★ OPTIMIZATION COMPLETED ★")
    print(f"Best Trial Score (Val PR-AUC): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key}: {formatted}")
    print("=" * 50)


if __name__ == "__main__":
    run_hpo()
