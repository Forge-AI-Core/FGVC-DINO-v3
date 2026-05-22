from pathlib import Path
from typing import Any
import argparse
import shutil
import yaml
import torch

from linear_head.get_data_loaders import get_data_loader
from linear_head.train_test_pipeline import (
    get_model,
    train_model,
    test_model,
    visualize_results,
    visualize_confusion_matrix,
)


def parse_cli_args() -> argparse.Namespace:
    """CLI 실행 인자를 파싱합니다.

    Returns:
        argparse.Namespace: 파싱된 인자 객체
    """
    parser = argparse.ArgumentParser(description="하이퍼파라미터 파서")
    parser.add_argument(
        "--hyperparams",
        type=str,
        default="hyperparams.yaml",
        help="하이퍼파라미터 파일 경로",
    )

    return parser.parse_args()


def load_hyperparams(file_path: Path) -> dict[str, Any]:
    """YAML 파일로부터 하이퍼파라미터를 로드합니다.

    Args:
        file_path (Path): 설정 파일 경로

    Returns:
        dict[str, Any]: 하이퍼파라미터 딕셔너리

    Raises:
        FileNotFoundError: 설정 파일이 존재하지 않을 때 발생
    """
    if not file_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {file_path}")

    with open(file=file_path, mode="r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)

    return hyperparams


def save_run_metadata(
    hyperparam_path: Path,
    model_name: str,
    dataset_name: str,
    best_metrics: dict[str, Any],
) -> None:
    """학습에 사용된 하이퍼파라미터 파일과 성능 지표를 결과 폴더에 저장합니다.

    Args:
        hyperparam_path (Path): 복사할 원본 하이퍼파라미터 파일 경로
        dataset_name (str): 데이터셋 이름
        best_metrics (dict[str, Any]): 최적 에포크의 지표 딕셔너리
    """
    results_dir = Path(f"results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. hyperparams.yaml 복사
    destination_yaml_path = results_dir / f"hyperparams.yaml"
    shutil.copy(src=hyperparam_path, dst=destination_yaml_path)

    print(f"Hyperparameters saved to: {destination_yaml_path}")

    # 2. 결과 지표 마크다운 생성 및 저장
    destination_md_path = results_dir / f"metrics_{dataset_name}.md"

    md_content = f"""# Training Metrics Summary ({model_name} on {dataset_name})

## Best Epoch: {best_metrics['epoch']}

| Metric | Value |
| :--- | :--- |
| Train Loss | {best_metrics['train_loss']:.4f} |
| Train Accuracy | {best_metrics['train_acc']:.2f}% |
| Test Loss | {best_metrics['test_loss']:.4f} |
| Test Accuracy | {best_metrics['test_acc']:.2f}% |
| Precision | {best_metrics['precision']:.4f} |
| Recall | {best_metrics['recall']:.4f} |
| F1 Score | {best_metrics['f1']:.4f} |
| MCC | {best_metrics['mcc']:.4f} |
| PR-AUC | {best_metrics['pr_auc']:.4f} |
| F-beta (0.5) | {best_metrics['fbeta']:.4f} |
"""
    with open(file=destination_md_path, mode="w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Metrics report saved to: {destination_md_path}")


def main(model_name: str, dataset_name: str) -> None:
    """main function \n
    Args:
        model_name: (vits16, vitb16, vitl16, vith16plus)\n
        dataset_name: (crops_0pct, crops_10pct, crops_25pct)\n
    Returns:
        None
    """
    args = parse_cli_args()
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(hyperparam_path)

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

    if dataset_name == "crops_0pct":
        dataset_dir = Path(hyperparams["data"]["CROPS_0PCT"]["DATASET_DIR"])
        dataset_name = hyperparams["data"]["CROPS_0PCT"]["CROPS_0PCT_NAME"]
    elif dataset_name == "crops_10pct":
        dataset_dir = Path(hyperparams["data"]["CROPS_10PCT"]["DATASET_DIR"])
        dataset_name = hyperparams["data"]["CROPS_10PCT"]["CROPS_10PCT_NAME"]
    elif dataset_name == "crops_25pct":
        dataset_dir = Path(hyperparams["data"]["CROPS_25PCT"]["DATASET_DIR"])
        dataset_name = hyperparams["data"]["CROPS_25PCT"]["CROPS_25PCT_NAME"]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    batch_size = hyperparams["data"]["BATCH_SIZE"]
    image_size = hyperparams["data"]["IMAGE_SIZE"]

    num_classes = hyperparams["model"]["NUM_CLASSES"]
    lora_rank = hyperparams["model"]["LORA_RANK"]
    lora_alpha = hyperparams["model"]["LORA_ALPHA"]
    target_modules = hyperparams["model"]["TARGET_MODULES"]

    num_epochs = hyperparams["train"]["NUM_EPOCHS"]
    learning_rate = float(hyperparams["train"]["LEARNING_RATE"])
    weight_decay = float(hyperparams["train"]["WEIGHT_DECAY"])
    early_stopping_patience = hyperparams["train"]["EARLY_STOPPING_PATIENCE"]
    checkpoint_dir = Path(hyperparams["train"]["CHECKPOINT_DIR"])

    train_loader, test_loader, _ = get_data_loader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(
        model_name=model_name,
        model_path=model_path,
        num_classes=num_classes,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs
    )

    # 기록용 딕셔너리
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_mcc": [],
        "test_pr_auc": [],
        "test_fbeta": [],
    }

    best_acc = 0.0
    patience_counter = 0
    best_metrics: dict[str, Any] = {}
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(
            device=device,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        # 에포크마다 성능 검증 및 얼리스탑핑 체크
        test_loss, test_acc, precision, recall, f1, mcc, pr_auc, fbeta = test_model(
            test_loader=test_loader,
            model=model,
            device=device,
            criterion=criterion,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} - Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}\n"
        )

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["test_precision"].append(precision)
        history["test_recall"].append(recall)
        history["test_f1"].append(f1)
        history["test_mcc"].append(mcc)
        history["test_pr_auc"].append(pr_auc)
        history["test_fbeta"].append(fbeta)

        lr_scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
                "pr_auc": pr_auc,
                "fbeta": fbeta,
            }
            # 최적 모델 가중치 저장
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(), checkpoint_dir / f"best_model_{model_name}.pth"
            )
        else:
            patience_counter += 1

            print(
                f"early stopping patience: {patience_counter}/{early_stopping_patience}\n"
            )
            if patience_counter >= early_stopping_patience:

                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    visualize_results(
        history=history,
        dataset_name=dataset_name,
        model_name=model_name,
    )

    # best 모델 가중치를 로드한 뒤 confusion matrix 생성
    best_model_path = checkpoint_dir / f"best_model_{model_name}.pth"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    visualize_confusion_matrix(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=test_loader.dataset.classes,
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


if __name__ == "__main__":
    model_name = (
        input("Enter model name (vits16, vitb16, vitl16, vith16plus): ").lower().strip()
    )
    dataset_name = (
        input("Enter dataset name (crops_0pct, crops_10pct, crops_25pct): ")
        .lower()
        .strip()
    )

    main(model_name=model_name, dataset_name=dataset_name)
