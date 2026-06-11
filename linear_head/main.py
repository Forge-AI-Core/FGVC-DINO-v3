from pathlib import Path
from typing import Any
import argparse
import shutil
import yaml
import torch

from linear_head.get_data_loaders import get_data_loader
from linear_head.train import train_model
from linear_head.validate import (
    validate_model,
    visualize_confusion_matrix,
    visualize_results,
)
from linear_head.test import test_model

from torch import nn
from peft import LoraConfig, get_peft_model


####################### #
# 하이퍼파라미터 로드 함수
####################### #
def load_hyperparams(file_path: Path) -> dict[str, Any]:
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)

    return hyperparams


################### #
# cli 인자 파싱 함수
################### #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINOv3 Standalone Inference Report Tester"
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default="hyperparams.yaml",
        help="설정 YAML 파일 경로",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vits16", "vitb16", "vitl16", "vith16plus"],
        help="사용할 모델 종류",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "vanilla_0pct",
            "vanilla_10pct",
            "vanilla_25pct",
            "unique_sampling_0pct",
            "unique_sampling_10pct",
            "unique_sampling_25pct",
        ],
        help="테스트할 데이터셋 이름",
    )

    return parser.parse_args()


########### #
# 모델 객체
########### #
def get_model(
    model_name: str,
    model_path: Path,
    num_classes: int,
    hidden_dim1: int,
    hidden_dim2: int,
    r: int = 16,
    lora_alpha: int = 16,
    target_modules: list = ["qkv"],
) -> nn.Module:
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=model_name,
        source="local",
        weights=str(model_path),
    )

    # 백본의 임베딩 차원 추출
    embed_dim = getattr(model, "embed_dim")

    # 분류기 헤드
    model.head = nn.Sequential(
        nn.Linear(embed_dim, hidden_dim1),
        nn.LayerNorm(hidden_dim1),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.LayerNorm(hidden_dim2),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(hidden_dim2, num_classes),
    )

    # LoRA 설정
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.5,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["head"],
    )

    peft_model = get_peft_model(
        model=model,
        peft_config=lora_config,
    )

    return peft_model


########################### #
# 학습 결과 metrics 저장 함수
########################### #
def save_run_metadata(
    hyperparam_path: Path,
    model_name: str,
    dataset_name: str,
    best_metrics: dict[str, Any],
) -> None:
    # 목적지
    results_dir = Path(f"results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # hyperparams.yaml 복사
    destination_yaml_path = results_dir / f"hyperparams.yaml"
    shutil.copy(src=hyperparam_path, dst=destination_yaml_path)

    print(f"Hyperparameters saved to: {destination_yaml_path}")

    # 결과 지표 마크다운 생성 및 저장
    destination_md_path = results_dir / f"metrics_{dataset_name}.md"

    # 작성될 마크다운 내용
    md_content = f"""# Training Metrics Summary ({model_name} on {dataset_name})

## Best Epoch: {best_metrics['epoch']}

| Metric | Value |
| :--- | :--- |
| Train Loss | {best_metrics['train_loss']:.4f} |
| Train Accuracy | {best_metrics['train_acc']:.2f}% |
| Val Loss | {best_metrics['val_loss']:.4f} |
| Val Accuracy | {best_metrics['val_acc']:.2f}% |
| Precision | {best_metrics['precision']:.4f} |
| Recall | {best_metrics['recall']:.4f} |
| F1 Score | {best_metrics['f1']:.4f} |
| MCC | {best_metrics['mcc']:.4f} |
| PR-AUC | {best_metrics['pr_auc']:.4f} |
| F-beta (0.5) | {best_metrics['fbeta']:.4f} |
"""
    # 저장
    with open(file=destination_md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Metrics report saved to: {destination_md_path}")


####################### #
# all process pipeline
####################### #
def pipeline(
    model_name: str,
    dataset_name: str,
    hyperparam_path: Path,
    optuna_override: dict[str, Any] | None = None,
) -> float:

    hyperparams = load_hyperparams(file_path=hyperparam_path)

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

    # checkpoint config variable
    checkpoint_dir = Path(hyperparams["test"]["linear_lora"]["CHECKPOINT_DIR"])
    checkpoint_path = checkpoint_dir / f"best_model_{dataset_name}_{model_name}.pth"

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

    # optuna config variable
    if optuna_override is not None:
        if "LORA_RANK" in optuna_override:
            lora_rank = optuna_override["LORA_RANK"]
        if "LORA_ALPHA" in optuna_override:
            lora_alpha = optuna_override["LORA_ALPHA"]
        if "LEARNING_RATE" in optuna_override:
            learning_rate = optuna_override["LEARNING_RATE"]
        if "WEIGHT_DECAY" in optuna_override:
            weight_decay = optuna_override["WEIGHT_DECAY"]
        if "HIDDEN_DIM1" in optuna_override:
            hidden_dim1 = optuna_override["HIDDEN_DIM1"]
        if "HIDDEN_DIM2" in optuna_override:
            hidden_dim2 = optuna_override["HIDDEN_DIM2"]

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
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    ).to(device)

    # loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs
    )

    ########################### #
    # train & validate process
    ########################### #
    # initialize history dictionary & metrics
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_mcc": [],
        "val_pr_auc": [],
        "val_fbeta": [],
    }
    best_pr_auc = 0.0
    patience_counter = 0
    best_metrics: dict[str, Any] = {}

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
        )

        # 검증
        val_loss, val_acc, precision, recall, f1, mcc, pr_auc, fbeta = validate_model(
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

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)
        history["val_mcc"].append(mcc)
        history["val_pr_auc"].append(pr_auc)
        history["val_fbeta"].append(fbeta)

        lr_scheduler.step()

        # PR-AUC 기준으로 베스트모델의 지표 작성
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            patience_counter = 0
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
                "pr_auc": pr_auc,
                "fbeta": fbeta,
            }
            # PR-AUC 기준으로 모델 가중치 파일 저장
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"best_model_{dataset_name}_{model_name}.pth",
            )

        else:
            patience_counter += 1
            print(
                f"early stopping patience: {patience_counter}/{early_stopping_patience}\n"
            )
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    ######## #
    # 시각화
    ######## #

    # history를 기반으로 수렴 과정 시각화
    visualize_results(
        history=history,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    # PR-AUC 기준으로 저장한 최적 모델 가중치를 로드한 뒤 confusion matrix 생성
    best_model_path = checkpoint_dir / f"best_model_{dataset_name}_{model_name}.pth"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    visualize_confusion_matrix(
        model=model,
        val_loader=val_loader,
        device=device,
        class_names=val_loader.dataset.classes,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    # 학습 메타데이터 및 지표 리포트 저장
    if best_metrics:
        save_run_metadata(
            hyperparam_path=hyperparam_path,
            model_name=model_name,
            dataset_name=dataset_name,
            best_metrics=best_metrics,
        )

    test_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        model_name=model_name,
        dataset_dir=dataset_dir,
        dataset_name=dataset_name,
        image_size=image_size,
    )

    return best_pr_auc


def main():
    # cli 인자 파싱
    args = parse_args()

    # 하이퍼 파라미터 로드
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(file_path=hyperparam_path)

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

    model_name = (
        input(f"Enter model name ({', '.join(model_options)}): ").lower().strip()
    )
    dataset_name = (
        input(f"Enter dataset name ({', '.join(dataset_options)}): ").lower().strip()
    )

    pipeline(
        model_name=model_name,
        dataset_name=dataset_name,
        hyperparam_path=hyperparam_path,
    )


if __name__ == "__main__":
    main()
