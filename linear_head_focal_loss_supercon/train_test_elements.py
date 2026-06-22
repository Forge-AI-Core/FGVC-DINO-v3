from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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

from linear_head_focal_loss_supercon.losses import FocalLoss, SupConLoss


########### #
# 모델 객체
########### #
class CustomDualHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim1, hidden_dim2, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim1),
            nn.LayerNorm(hidden_dim1),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim1, hidden_dim2),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim2, num_classes),
        )
        self.return_embeddings = False

    def forward(self, x):
        feats = self.features(x)
        if self.return_embeddings:
            return F.normalize(feats, dim=1)
        else:
            return self.classifier(feats)


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
    """dino 모델 로드 및 LoRA 설정 함수"""
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=model_name,
        source="local",
        weights=str(model_path),
    )

    embed_dim = getattr(model, "embed_dim")

    # CustomDualHead로 교체
    model.head = CustomDualHead(embed_dim, hidden_dim1, hidden_dim2, num_classes)

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
):
    """데이터로더와 모델만 반환하도록 수정 (Optimizer/Scheduler는 각 Stage에서 생성)"""
    train_loader, val_loader, test_loader = get_data_loader(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    return train_loader, val_loader, test_loader, device, model


########################### #
# 2-Stage Process
########################### #
def freeze_for_stage2(model: nn.Module):
    """Stage 2를 위해 분류기의 최종 Linear를 제외한 모든 파라미터 동결"""
    for param in model.parameters():
        param.requires_grad = False

    # CustomDualHead의 classifier만 열어줌
    if hasattr(model, "base_model"):
        base = model.base_model.model
        if hasattr(base, "head"):
            if hasattr(base.head, "modules_to_save"):
                for adapter in base.head.modules_to_save.values():
                    for param in adapter.classifier.parameters():
                        param.requires_grad = True
                for param in base.head.original_module.classifier.parameters():
                    param.requires_grad = True
            else:
                for param in base.head.classifier.parameters():
                    param.requires_grad = True


def set_return_embeddings(model: nn.Module, val: bool):
    """PEFT의 ModulesToSaveWrapper 내부의 실제 CustomDualHead에 속성을 설정합니다."""
    if hasattr(model, "base_model"):
        head = getattr(model.base_model.model, "head", None)
        if head is not None:
            if hasattr(head, "modules_to_save"):
                for adapter_name in head.modules_to_save:
                    head.modules_to_save[adapter_name].return_embeddings = val
                head.original_module.return_embeddings = val
            else:
                head.return_embeddings = val


class _TwoViewDataset(torch.utils.data.Dataset):
    """같은 이미지에 서로 다른 augmentation을 2회 적용하여 (view1, view2, label) 반환.

    SupConLoss는 [bsz, n_views, dim] 형태를 요구하며, n_views≥2일 때
    동일 이미지의 서로 다른 뷰를 양성 쌍(positive pair)으로 활용하여
    contrastive learning이 정상적으로 작동합니다.
    """

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        from PIL import Image

        img = Image.open(path).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label


def run_stage1_supcon(
    num_epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    accumulation_steps: int,
    checkpoint_path: Path,
    stage1_batch_size: int = None,
    early_stopping_patience: int = 15,
) -> None:
    print("\n" + "=" * 50)
    print("▶ Stage 1: Representation Learning (SupCon)")
    print("=" * 50)

    # 임베딩 반환 모드 설정
    set_return_embeddings(model, True)

    if stage1_batch_size is None:
        stage1_batch_size = train_loader.batch_size

    # SupCon 전용 2-view augmentation DataLoader 생성
    from torchvision import transforms
    from linear_head.get_data_loaders import SquarePad

    image_size = train_loader.dataset[0][0].shape[-1]
    supcon_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, scale=(0.8, 1.2)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    supcon_train_dataset = _TwoViewDataset(train_loader.dataset, supcon_transform)
    supcon_train_loader = DataLoader(
        dataset=supcon_train_dataset,
        batch_size=stage1_batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )

    supcon_val_dataset = _TwoViewDataset(val_loader.dataset, supcon_transform)
    supcon_val_loader = DataLoader(
        dataset=supcon_val_dataset,
        batch_size=stage1_batch_size,
        shuffle=True,  # SupConLoss 검증을 위해 반드시 섞여야 함 (안 섞이면 배치 내 단일 클래스로 인해 Loss 망가짐)
        num_workers=6,
        pin_memory=True,
    )

    criterion = SupConLoss(temperature=0.07).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            supcon_train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Stage1 Train]",
        )
        for index, (view1, view2, labels) in enumerate(progress_bar):
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                feat1 = model(view1)
                feat2 = model(view2)
                # [bsz, 2, dim] — SupConLoss의 핵심 입력 형태
                features = torch.stack([feat1, feat2], dim=1)
                loss = criterion(features, labels)

            (loss / accumulation_steps).backward()

            if (index + 1) % accumulation_steps == 0 or (index + 1) == len(
                supcon_train_loader
            ):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{running_loss/(index+1):.4f}"})

        train_loss = running_loss / len(supcon_train_loader)

        # Validation for Stage 1
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for view1, view2, labels in supcon_val_loader:
                view1 = view1.to(device)
                view2 = view2.to(device)
                labels = labels.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    feat1 = model(view1)
                    feat2 = model(view2)
                    features = torch.stack([feat1, feat2], dim=1)
                    loss = criterion(features, labels)
                val_loss += loss.item()
        val_loss /= len(supcon_val_loader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Stage 1 Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}\n"
        )
        lr_scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Best Model Saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"  --> Early Stopping Patience: {patience_counter}/{early_stopping_patience}"
            )
            if patience_counter >= early_stopping_patience:
                print(f"\n[Stage 1] Early stopping triggered at epoch {epoch+1}.")
                break


def run_stage2_focal(
    num_epochs: int,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    accumulation_steps: int,
    early_stopping_patience: int,
    checkpoint_path: Path,
    dataset_name: str,
    model_name: str,
    hyperparam_path: Path,
) -> float:
    print("\n" + "=" * 50)
    print("▶ Stage 2: Classifier Fine-Tuning (Alpha-Focal)")
    print("=" * 50)

    # 로짓 반환 모드 설정
    set_return_embeddings(model, False)

    criterion = FocalLoss(alpha=[0.2, 0.6, 0.2], gamma=2.0).to(device)

    # Stage 2 Optimizer: Only classifier parameters
    stage2_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        stage2_params, lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

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
    best_metrics = {}

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
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            with open(checkpoint_path.parent / f"best_val_threshold_90_{dataset_name}_{model_name}.txt", "w") as f:
                f.write(str(val_threshold_at_90))
        else:
            patience_counter += 1
            print(
                f"early stopping patience: {patience_counter}/{early_stopping_patience}\n"
            )
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

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
        val_threshold_at_90=best_metrics.get("val_threshold_at_90", None) if best_metrics else None,
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
        val_threshold_at_90=val_threshold_at_90,
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
