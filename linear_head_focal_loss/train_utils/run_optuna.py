from pathlib import Path
from typing import Any
import optuna

import torch

# 기존 trainer.py의 학습 함수 임포트
from linear_head_focal_loss.main import load_hyperparams, parse_args
from linear_head_focal_loss.train_test_elements import get_model
from linear_head_focal_loss.get_data_loaders import get_data_loader
from linear_head_focal_loss.train import train_model
from linear_head_focal_loss.validate import validate_model

args = parse_args()
hyperparams = load_hyperparams(file_path=args.hyperparams)
model_name = input("Enter model name (vits16, vitb16, vitl16, vith16plus): ")
dataset_name = input(
    "Enter dataset name (vanilla_0pct, vanilla_10pct, vanilla_25pct, unique_sampling_0pct, unique_sampling_10pct, unique_sampling_25pct): "
)

# model config variable
if model_name == "vits16":
    model_path = Path(hyperparams["model"]["vits16"]["PATH"])
    model_name = hyperparams["model"]["vits16"]["NAME"]
elif model_name == "vitb16":
    model_path = Path(hyperparams["model"]["vitb16"]["PATH"])
    model_name = hyperparams["model"]["vitb16"]["NAME"]
elif model_name == "vitl16":
    model_path = Path(hyperparams["model"]["vitl16"]["PATH"])
    model_name = hyperparams["model"]["vitl16"]["NAME"]
elif model_name == "vith16plus":
    model_path = Path(hyperparams["model"]["vith16plus"]["PATH"])
    model_name = hyperparams["model"]["vith16plus"]["NAME"]
else:
    raise ValueError(f"Invalid model name: {model_name}")

# dataset config variable
if dataset_name == "vanilla_0pct":
    dataset_dir = Path(hyperparams["data"]["vanilla_0pct"]["DATASET_DIR"])
    dataset_name = hyperparams["data"]["vanilla_0pct"]["DATASET_NAME"]
elif dataset_name == "vanilla_10pct":
    dataset_dir = Path(hyperparams["data"]["vanilla_10pct"]["DATASET_DIR"])
    dataset_name = hyperparams["data"]["vanilla_10pct"]["DATASET_NAME"]
elif dataset_name == "vanilla_25pct":
    dataset_dir = Path(hyperparams["data"]["vanilla_25pct"]["DATASET_DIR"])
    dataset_name = hyperparams["data"]["vanilla_25pct"]["DATASET_NAME"]
elif dataset_name == "unique_sampling_0pct":
    dataset_dir = Path(hyperparams["data"]["unique_sampling_0pct"]["DATASET_DIR"])
    dataset_name = hyperparams["data"]["unique_sampling_0pct"]["DATASET_NAME"]
elif dataset_name == "unique_sampling_10pct":
    dataset_dir = Path(hyperparams["data"]["unique_sampling_10pct"]["DATASET_DIR"])
    dataset_name = hyperparams["data"]["unique_sampling_10pct"]["DATASET_NAME"]
elif dataset_name == "unique_sampling_25pct":
    dataset_dir = Path(hyperparams["data"]["unique_sampling_25pct"]["DATASET_DIR"])
    dataset_name = hyperparams["data"]["unique_sampling_25pct"]["DATASET_NAME"]
else:
    raise ValueError(f"Invalid dataset name: {dataset_name}")

################### #
# hyper parameters
################### #
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


####################### #
# all process pipeline
####################### #
def pipeline(
    optuna_override: dict[str, Any] | None = None,
) -> float:
    # optuna config variable (로컬 변수로 바인딩하여 scope 에러 방지)
    override = optuna_override or {}
    lora_rank_val = override.get("LORA_RANK", lora_rank)
    lora_alpha_val = override.get("LORA_ALPHA", lora_alpha)
    learning_rate_val = override.get("LEARNING_RATE", learning_rate)
    weight_decay_val = override.get("WEIGHT_DECAY", weight_decay)
    hidden_dim1_val = override.get("HIDDEN_DIM1", hidden_dim1)
    hidden_dim2_val = override.get("HIDDEN_DIM2", hidden_dim2)

    ########################################################## #
    # 데이터 로더, 디바이스, 모델, 손실함수, 옵티마이저, 스케줄러 정의
    ########################################################## #
    # dataloader
    train_loader, val_loader, _ = get_data_loader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on device: {device}")

    # model
    model = get_model(
        model_name=model_name,
        model_path=model_path,
        num_classes=num_classes,
        hidden_dim1=hidden_dim1_val,
        hidden_dim2=hidden_dim2_val,
        r=lora_rank_val,
        lora_alpha=lora_alpha_val,
        target_modules=target_modules,
    ).to(device)

    # loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate_val, weight_decay=weight_decay_val
    )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs
    )

    ########################### #
    # train & validate process
    ########################### #
    # initialize metrics
    best_pr_auc = 0.0
    patience_counter = 0

    # epochs
    for epoch in range(num_epochs):
        # 학습
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

        # 검증
        val_loss, val_acc, _, _, _, _, pr_auc, _ = validate_model(
            val_loader=val_loader,
            model=model,
            device=device,
            criterion=criterion,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} - Val PR-AUC: {pr_auc:.4f}\n"
        )

        lr_scheduler.step()

        # PR-AUC 기준으로 베스트모델의 지표 작성
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0

        else:
            patience_counter += 1
            print(
                f"early stopping patience: {patience_counter}/{early_stopping_patience}\n"
            )
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_pr_auc


def objective(trial: optuna.Trial) -> float:
    """
    Optuna Trial별로 제안된 하이퍼파라미터를 적용하여 학습을 수행하고 결과를 반환합니다.
    Args:
        trial (optuna.Trial): 하이퍼파라미터 제안용 객체
    Returns:
        float: 최고 검증 PR-AUC (Val PR-AUC, %)
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
    best_pr_auc: float = pipeline(
        optuna_override={
            "LORA_RANK": lora_rank,
            "LORA_ALPHA": lora_alpha,
            "LEARNING_RATE": learning_rate,
            "WEIGHT_DECAY": weight_decay,
        },
    )

    return best_pr_auc


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
    print(f"Best Trial Score (Val PR-AUC): {study.best_value:.4f}")
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
