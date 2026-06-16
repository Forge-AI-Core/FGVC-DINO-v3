"""Optuna HPO for 2-Stage SupCon + Focal Loss pipeline.

핵심 탐색 파라미터: learning_rate, weight_decay, lora_rank, lora_alpha.
해당 폴더(linear_head_focal_loss_supercon)의 2-Stage 학습 파이프라인을 그대로 사용합니다.

Why: 기존 run_optuna.py는 linear_head의 단일 CrossEntropyLoss 루프를 사용했으나,
이 모듈은 Stage1(SupCon) → Stage2(FocalLoss) 구조이므로 파이프라인을 올바르게 반영해야 합니다.
"""

from pathlib import Path
from typing import Any
import sys

# 프로젝트 루트(FGVC-DINO-v3)를 sys.path에 추가하여 모듈을 찾을 수 있게 합니다.
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import optuna
import torch

from linear_head_focal_loss_supercon.main import load_hyperparams
from linear_head_focal_loss_supercon.train_test_elements import (
    get_modules,
    run_stage1_supcon,
    run_stage2_focal,
    freeze_for_stage2,
)


# ──────────────────────────────────────────────
# 설정 로드 (모듈 로드 시 input 호출 방지)
# ──────────────────────────────────────────────
HYPERPARAMS_PATH = Path("hyperparams.yaml")


def _resolve_config(
    hyperparams: dict[str, Any],
) -> tuple[str, Path, str, Path]:
    """설정 파일로부터 모델/데이터셋 선택지를 파싱하고 사용자 입력을 받아 config를 반환합니다.

    Returns:
        (model_name, model_path, dataset_name, dataset_dir)
    """
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

    model_key = input(f"Enter model name ({', '.join(model_options)}): ").lower().strip()
    dataset_key = input(f"Enter dataset name ({', '.join(dataset_options)}): ").lower().strip()

    model_config = hyperparams["model"][model_key]
    dataset_config = hyperparams["data"][dataset_key]

    return (
        model_config["NAME"],
        Path(model_config["PATH"]),
        dataset_config["DATASET_NAME"],
        Path(dataset_config["DATASET_DIR"]),
    )


# ──────────────────────────────────────────────
# 2-Stage 학습 파이프라인
# ──────────────────────────────────────────────
def _run_two_stage_pipeline(
    hyperparams: dict[str, Any],
    model_name: str,
    model_path: Path,
    dataset_name: str,
    dataset_dir: Path,
    learning_rate: float,
    weight_decay: float,
    lora_rank: int,
    lora_alpha: int,
) -> float:
    """2-Stage(SupCon→Focal) 학습을 수행하고 best PR-AUC를 반환합니다.

    Args:
        hyperparams: 전체 하이퍼파라미터 딕셔너리
        model_name: DINOv3 모델 이름
        model_path: 백본 가중치 경로
        dataset_name: 데이터셋 이름
        dataset_dir: 데이터셋 디렉토리 경로
        learning_rate: Optuna가 제안한 학습률
        weight_decay: Optuna가 제안한 가중치 감쇄
        lora_rank: Optuna가 제안한 LoRA Rank
        lora_alpha: Optuna가 제안한 LoRA Alpha

    Returns:
        Stage 2에서 달성한 최고 Danger PR-AUC 값
    """
    batch_size = hyperparams["data"]["BATCH_SIZE"]
    image_size = hyperparams["data"]["IMAGE_SIZE"]
    num_classes = hyperparams["model"]["NUM_CLASSES"]
    hidden_dim1 = hyperparams["model"]["HIDDEN_DIM1"]
    hidden_dim2 = hyperparams["model"]["HIDDEN_DIM2"]
    target_modules = hyperparams["model"]["TARGET_MODULES"]

    num_epochs = hyperparams["train"]["NUM_EPOCHS"]
    early_stopping_patience = hyperparams["train"]["EARLY_STOPPING_PATIENCE"]
    accumulation_steps = hyperparams["train"].get("ACCUMULATION_STEPS", 1)
    stage1_batch_size = hyperparams["train"].get("STAGE1_BATCH_SIZE", batch_size)

    checkpoint_dir = Path(hyperparams["test"]["linear_lora"]["CHECKPOINT_DIR"])

    # 모듈 초기화
    train_loader, val_loader, _, device, model = get_modules(
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

    # 에폭 50:50 분할
    num_epochs_stage1 = num_epochs // 2
    num_epochs_stage2 = num_epochs - num_epochs_stage1

    # Stage 1: SupCon
    checkpoint_stage1 = checkpoint_dir / f"optuna_stage1_{dataset_name}_{model_name}.pth"
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

    # Load Best Stage 1 & Freeze
    if checkpoint_stage1.exists():
        model.load_state_dict(torch.load(checkpoint_stage1, weights_only=True))
    freeze_for_stage2(model)

    # Stage 2: Focal Loss
    checkpoint_stage2 = checkpoint_dir / f"optuna_stage2_{dataset_name}_{model_name}.pth"
    best_pr_auc = run_stage2_focal(
        num_epochs=num_epochs_stage2,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        accumulation_steps=accumulation_steps,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_stage2,
        dataset_name=dataset_name,
        model_name=model_name,
        hyperparam_path=HYPERPARAMS_PATH,
    )

    # VRAM 해제
    del model
    torch.cuda.empty_cache()

    return best_pr_auc


# ──────────────────────────────────────────────
# Optuna Objective
# ──────────────────────────────────────────────
def _create_objective(
    hyperparams: dict[str, Any],
    model_name: str,
    model_path: Path,
    dataset_name: str,
    dataset_dir: Path,
):
    """클로저로 objective 함수를 생성하여 모듈 레벨 전역 변수 의존을 제거합니다."""

    def objective(trial: optuna.Trial) -> float:
        """Optuna Trial에서 핵심 하이퍼파라미터를 제안받아 2-Stage 학습을 수행합니다.

        Args:
            trial: Optuna Trial 객체

        Returns:
            최고 Danger PR-AUC 값
        """
        # 핵심 하이퍼파라미터 탐색
        lora_rank: int = trial.suggest_categorical("lora_rank", [4, 8, 16])
        lora_alpha: int = lora_rank * 2

        learning_rate: float = trial.suggest_float(
            "learning_rate", 1e-5, 1e-4, log=True,
        )
        weight_decay: float = trial.suggest_float(
            "weight_decay", 1e-3, 5e-2, log=True,
        )

        print("\n" + "=" * 50)
        print(f"[Optuna Trial #{trial.number}]")
        print(f"  LORA_RANK     : {lora_rank}")
        print(f"  LORA_ALPHA    : {lora_alpha}")
        print(f"  LEARNING_RATE : {learning_rate:.6f}")
        print(f"  WEIGHT_DECAY  : {weight_decay:.5f}")
        print("=" * 50 + "\n")

        best_pr_auc = _run_two_stage_pipeline(
            hyperparams=hyperparams,
            model_name=model_name,
            model_path=model_path,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        return best_pr_auc

    return objective


# ──────────────────────────────────────────────
# Study Runner
# ──────────────────────────────────────────────
N_TRIALS = 20
TPE_SEED = 2026


def run_hpo(n_trials: int = N_TRIALS) -> None:
    """하이퍼파라미터 최적화 Study를 생성하고 탐색을 실행합니다.

    Args:
        n_trials: 실험 횟수
    """
    hyperparams = load_hyperparams(file_path=HYPERPARAMS_PATH)
    model_name, model_path, dataset_name, dataset_dir = _resolve_config(hyperparams)

    objective = _create_objective(
        hyperparams=hyperparams,
        model_name=model_name,
        model_path=model_path,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
    )

    study = optuna.create_study(
        study_name="dinov3_supercon_focal_hpo",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TPE_SEED),
    )

    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 50)
    print("★ OPTIMIZATION COMPLETED ★")
    print(f"Best Trial Score (Danger PR-AUC): {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
        print(f"  {key}: {formatted}")
    print("=" * 50)


if __name__ == "__main__":
    run_hpo()
