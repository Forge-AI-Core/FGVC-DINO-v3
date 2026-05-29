"""Optuna를 이용한 DINOv3 PEFT 분류 파이프라인 하이퍼파라미터 최적화(HPO) 스크립트.

이 스크립트는 LORA_RANK(및 비례하는 ALPHA), LEARNING_RATE, WEIGHT_DECAY의
최적 조합을 탐색하여 검증 정확도(Validation Accuracy)를 극대화합니다.
"""

from pathlib import Path
from typing import Any
import optuna

# 기존 main.py의 학습 함수 임포트
from linear_head.main import main as train_pipeline


def objective(trial: optuna.Trial) -> float:
    """
    Optuna Trial별로 제안된 하이퍼파라미터를 적용하여 학습을 수행하고 결과를 반환합니다.

    Args:
        trial (optuna.Trial): 하이퍼파라미터 제안용 객체

    Returns:
        float: 최고 검증 정확도 (Val Accuracy, %)
    """
    # 1. 탐색할 하이퍼파라미터 정의
    # LORA_RANK 후보군 설정 (가용한 물리 자원 및 가중치 안정성 고려)
    lora_rank: int = trial.suggest_categorical("lora_rank", [8, 16])

    # LORA_ALPHA는 RANK에 비례하여 동기화 (업데이트 스케일 factor 일치)
    lora_alpha: int = lora_rank * 2

    # LEARNING_RATE 범위를 로그 스케일로 정밀 탐색 (1e-5 ~ 3e-4)
    learning_rate: float = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)

    # WEIGHT_DECAY 범위를 로그 스케일로 상향 탐색 (과적합 격차 좁히기 목적)
    weight_decay: float = trial.suggest_float("weight_decay", 1e-3, 5e-2, log=True)

    print("\n" + "=" * 50)
    print(f"[Optuna Trial #{trial.number}]")
    print(f" - Suggested LORA_RANK   : {lora_rank}")
    print(f" - Suggested LORA_ALPHA  : {lora_alpha}")
    print(f" - Suggested L_RATE      : {learning_rate:.6f}")
    print(f" - Suggested W_DECAY     : {weight_decay:.5f}")
    print("=" * 50 + "\n")

    # 2. 오버라이딩 딕셔너리 구축하여 학습 파이프라인 호출
    best_val_acc: float = train_pipeline(
        model_name="vits16",
        dataset_name="unique_sampling_25pct",
        hyperparam_path=Path("hyperparams.yaml"),
        optuna_override={
            "LORA_RANK": lora_rank,
            "LORA_ALPHA": lora_alpha,
            "LEARNING_RATE": learning_rate,
            "WEIGHT_DECAY": weight_decay,
        },
    )

    return best_val_acc


def run_hpo(n_trials: int) -> None:
    """
    하이퍼파라미터 최적화 Study를 생성하고 탐색을 진행합니다.

    Args:
        n_trials (int): 시도할 총 실험 횟수 (기본값: 15회)
    """
    # 목적 함수 방향을 '최대화(maximize)'로 설정
    study = optuna.create_study(
        study_name="dinov3_peft_optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2026),  # 재현성을 위해 시드 고정
    )

    study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 50)
    print("★ OPTIMIZATION COMPLETED ★")
    print(f"Best Trial Score (Val Acc): {study.best_value:.2f}%")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        if key == "learning_rate" or key == "weight_decay":
            print(f" - {key}: {value:.6f}")
        else:
            print(f" - {key}: {value}")
    print("=" * 50)


if __name__ == "__main__":
    # 총 15회 탐색 실행 (소형 모델 기준 빠른 탐색을 위해 15~20회 권장)
    run_hpo(n_trials=20)
