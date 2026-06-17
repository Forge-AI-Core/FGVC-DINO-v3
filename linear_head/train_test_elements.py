from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from linear_head.get_data_loaders import get_data_loader

from linear_head.train import train_model
from linear_head.validate import validate_model
from linear_head.test import test_model, test_model_with_samples

from linear_head.train_utils.train_metrics_utils import (
    visualize_val_results,
    visualize_val_confusion_matrix,
    save_val_run_metadata,
)
from linear_head.test_utils.test_metrics_utils import (
    visualize_test_confusion_matrix,
    save_run_test_metadata,
)
from linear_head.test_utils.test_metrics_utils_for_samples import (
    generate_and_save_pr_curve,
)


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
    head_dropout: float = 0.5,
    lora_dropout: float = 0.5,
) -> nn.Module:
    """dino 모델 로드 및 LoRA 설정 함수"""
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
        nn.Dropout(p=head_dropout),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.LayerNorm(hidden_dim2),
        nn.GELU(),
        nn.Dropout(p=head_dropout),
        nn.Linear(hidden_dim2, num_classes),
    )

    # LoRA 설정
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["head"],
    )

    peft_model = get_peft_model(
        model=model,
        peft_config=lora_config,
    )

    return peft_model


def get_modules(
    dataset_dir: Path,
    batch_size: int,
    image_size: int,
    model_name: str,
    model_path: Path,
    num_classes: int,
    hidden_dim1: int,
    hidden_dim2: int,
    lora_rank: int,
    lora_alpha: int,
    target_modules: list,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    accumulation_steps: int,
    head_dropout: float = 0.5,
    lora_dropout: float = 0.5,
    label_smoothing: float = 0.0,
):
    """학습/검증/테스트에 필요한 모듈(데이터로더, 모델, 옵티마이저 등) 초기화 함수"""
    # dataloader
    train_loader, val_loader, test_loader = get_data_loader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        head_dropout=head_dropout,
        lora_dropout=lora_dropout,
    ).to(device)

    # loss function
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        device,
        model,
        criterion,
        optimizer,
        lr_scheduler,
    )


########################### #
# train & validate process
########################### #
def run_train_val_process(
    num_epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    accumulation_steps: int,
    early_stopping_patience: int,
    checkpoint_path: Path,
    dataset_name: str,
    model_name: str,
    hyperparam_path: Path,
) -> float:
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
    checkpoint_dir = checkpoint_path.parent

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
            danger_precision,
            danger_recall,
            danger_f1,
            mcc,
            danger_pr_auc,
            danger_fbeta,
            val_threshold_at_90,
            val_recall_at_90,
        ) = validate_model(
            val_loader=val_loader,
            model=model,
            device=device,
            criterion=criterion,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} - Val PR-AUC: {danger_pr_auc:.4f}\n"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(danger_precision)
        history["val_recall"].append(danger_recall)
        history["val_f1"].append(danger_f1)
        history["val_mcc"].append(mcc)
        history["val_pr_auc"].append(danger_pr_auc)
        history["val_fbeta"].append(danger_fbeta)

        lr_scheduler.step()

        if danger_pr_auc > best_pr_auc:
            best_pr_auc = danger_pr_auc
            patience_counter = 0
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "precision": danger_precision,
                "recall": danger_recall,
                "f1": danger_f1,
                "mcc": mcc,
                "pr_auc": danger_pr_auc,
                "fbeta": danger_fbeta,
                "val_threshold_at_90": val_threshold_at_90,
                "val_recall_at_90": val_recall_at_90,
            }
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                model.state_dict(),
                checkpoint_path,
            )
            # Save the threshold as a text file
            with open(checkpoint_dir / f"best_val_threshold_90_{dataset_name}_{model_name}.txt", "w") as f:
                f.write(str(val_threshold_at_90))
        else:
            patience_counter += 1
            print(
                f"early stopping patience: {patience_counter}/{early_stopping_patience}\n"
            )
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 결과물 저장 경로
    results_dir = Path(f"results/{model_name}/{dataset_name}/val")
    results_dir.mkdir(parents=True, exist_ok=True)

    visualize_val_results(
        history=history,
        dataset_name=dataset_name,
        model_name=model_name,
        results_dir=results_dir,
    )

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    visualize_val_confusion_matrix(
        model=model,
        val_loader=val_loader,
        device=device,
        class_names=val_loader.dataset.classes,
        dataset_name=dataset_name,
        model_name=model_name,
        results_dir=results_dir,
    )

    pr_save_path = results_dir / f"pr_curve_{dataset_name}.png"
    generate_and_save_pr_curve(
        model=model,
        test_loader=val_loader,
        device=device,
        class_names=val_loader.dataset.classes,
        save_path=pr_save_path,
    )

    if best_metrics:
        save_val_run_metadata(
            hyperparam_path=hyperparam_path,
            model_name=model_name,
            dataset_name=dataset_name,
            best_metrics=best_metrics,
            results_dir=results_dir,
        )

    return best_pr_auc


########################### #
# test process for testset
########################### #
def run_testset_process(
    checkpoint_path: Path,
    device: torch.device,
    model: nn.Module,
    model_name: str,
    dataset_name: str,
    image_size: int,
    test_loader: DataLoader,
    hyperparam_path: Path,
) -> None:
    (
        test_acc,
        danger_precision,
        danger_recall,
        danger_f1,
        mcc,
        danger_pr_auc,
        danger_fbeta,
        val_threshold_at_90,
    ) = test_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        model_name=model_name,
        dataset_name=dataset_name,
        image_size=image_size,
        test_loader=test_loader,
        hyperparam_path=hyperparam_path,
    )

    # 결과물 저장 경로
    results_dir = Path(f"results/{model_name}/{dataset_name}/test")
    results_dir.mkdir(parents=True, exist_ok=True)

    visualize_test_confusion_matrix(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=test_loader.dataset.classes,
        dataset_name=dataset_name,
        model_name=model_name,
        results_dir=results_dir,
    )

    save_run_test_metadata(
        hyperparam_path=hyperparam_path,
        model_name=model_name,
        dataset_name=dataset_name,
        test_metrics={
            "test_acc": test_acc,
            "precision": danger_precision,
            "recall": danger_recall,
            "f1": danger_f1,
            "mcc": mcc,
            "pr_auc": danger_pr_auc,
            "fbeta": danger_fbeta,
            "val_threshold_at_90": val_threshold_at_90,
        },
        results_dir=results_dir,
    )


########################### #
# test process for samples
########################### #
def run_sample_test_process(
    model_name: str,
    test_loader: DataLoader,
    dataset_name: str,
    image_size: int,
    checkpoint_path: Path,
    device: torch.device,
    model: nn.Module,
    patch_size: int,
) -> None:
    # 결과물 저장 경로
    results_dir = Path(f"results/{model_name}/{dataset_name}/test/with_samples")
    results_dir.mkdir(parents=True, exist_ok=True)

    test_model_with_samples(
        model_name=model_name,
        test_loader=test_loader,
        dataset_name=dataset_name,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        device=device,
        model=model,
        patch_size=patch_size,
        results_dir=results_dir,
        visualize_samples=True,
    )
