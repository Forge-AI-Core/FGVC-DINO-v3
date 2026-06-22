import torch

from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    precision_recall_curve,
)
import numpy as np
import matplotlib.pyplot as plt


########### #
# 검증 함수
########### #
def validate_model(
    val_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    model.eval()

    # initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    avg_loss = 0.0
    val_acc = 0.0
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    # validation loop
    with torch.inference_mode():
        progress_bar = tqdm(iterable=val_loader, desc="val")
        for index, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            # loss
            running_loss += loss.item()
            avg_loss = running_loss / (index + 1)

            # accuracy
            predictions = logits.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
            val_acc = 100.0 * correct / total if total > 0 else 0.0

            # postfix
            progress_bar.set_postfix(
                {"avg_loss": f"{avg_loss:.4f}", "val_acc": f"{val_acc:.2f}%"}
            )

            # 메트릭 계산을 위해 예측값, 실제값, 확률값 저장
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1).cpu().numpy())

        print(
            f"epoch {epoch+1}/{num_epochs}, val loss: {avg_loss:.4f}, val acc: {val_acc:.2f}%"
        )

        # precision, recall, f1_score
        danger_precision = precision_score(
            y_true=all_labels,
            y_pred=all_predictions,
            average=None,
            zero_division=0,
        )[1]
        danger_recall = recall_score(
            y_true=all_labels,
            y_pred=all_predictions,
            average=None,
            zero_division=0,
        )[1]
        danger_f1 = f1_score(
            y_true=all_labels,
            y_pred=all_predictions,
            average=None,
            zero_division=0,
        )[1]

        mcc = matthews_corrcoef(y_true=all_labels, y_pred=all_predictions)

        # PR-AUC (one-vs-rest)
        all_probs_np = np.array(all_probs)
        num_classes = all_probs_np.shape[1]
        all_labels_onehot = np.eye(num_classes)[np.array(all_labels)]
        danger_pr_auc = average_precision_score(
            y_true=all_labels_onehot,
            y_score=all_probs_np,
            average=None,
        )[1]

        # F-beta (beta=0.5, precision 중시)
        danger_fbeta = fbeta_score(
            y_true=all_labels,
            y_pred=all_predictions,
            beta=0.5,
            average=None,
            zero_division=0,
        )[1]

        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(
            y_true=all_labels_onehot[:, 1], y_score=all_probs_np[:, 1]
        )

        target_precision = 0.90
        idx = np.where(precisions >= target_precision)[0]
        if len(idx) > 0:
            best_idx = idx[0]
            val_threshold_at_90 = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
            val_recall_at_90 = recalls[best_idx]
        else:
            val_threshold_at_90 = 1.0
            val_recall_at_90 = 0.0

        print(
            f"Danger Precision: {danger_precision:.4f}, Danger Recall: {danger_recall:.4f}, Danger F1: {danger_f1:.4f}, "
            f"MCC: {mcc:.4f}, Danger PR-AUC: {danger_pr_auc:.4f}, Danger F-beta: {danger_fbeta:.4f}\n"
            f"Val Threshold @ 0.90 Precision: {val_threshold_at_90:.4f}, Val Recall: {val_recall_at_90:.4f}"
        )

    return avg_loss, val_acc, danger_precision, danger_recall, danger_f1, mcc, danger_pr_auc, danger_fbeta, val_threshold_at_90, val_recall_at_90


######################################## #
# validation 결과에 대한 그래프 시각화 함수
######################################## #
def visualize_results(
    history: dict[str, list[float]], dataset_name: str, model_name: str
) -> None:
    """학습 결과(Loss, Acc, F1, P&R)를 시각화하여 저장합니다."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 15))
    plt.suptitle(f"Linear-Head DINOv3 Training Results ({dataset_name})", fontsize=16)

    # 1. Loss (Train & Val)
    plt.subplot(3, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="x")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # 2. Accuracy (Train & Val)
    plt.subplot(3, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
    plt.plot(epochs, history["val_acc"], color="green", label="Val Acc", marker="s")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    # 3. MCC (Matthews Correlation Coefficient)
    plt.subplot(3, 2, 3)
    plt.plot(epochs, history["val_mcc"], color="purple", label="MCC", marker="d")
    plt.title("Matthews Correlation Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("MCC")
    plt.legend()
    plt.grid(True)

    # 4. Precision, Recall & F1
    plt.subplot(3, 2, 4)
    plt.plot(epochs, history["val_precision"], label="Danger Precision", marker="^")
    plt.plot(epochs, history["val_recall"], label="Danger Recall", marker="v")
    plt.plot(epochs, history["val_f1"], color="red", label="Danger F1", marker="d")
    plt.title("Danger Precision, Recall & F1 (Val)")
    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # 5. PR-AUC (Average Precision)
    plt.subplot(3, 2, 5)
    plt.plot(epochs, history["val_pr_auc"], color="orange", label="Danger PR-AUC", marker="s")
    plt.title("Danger PR-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR-AUC")
    plt.legend()
    plt.grid(True)

    # 6. F-beta (beta=0.5)
    plt.subplot(3, 2, 6)
    plt.plot(
        epochs,
        history["val_fbeta"],
        color="teal",
        label="Danger F\u03b2 (\u03b2=0.5)",
        marker="p",
    )
    plt.title("Danger F-beta Score (\u03b2=0.5)")
    plt.xlabel("Epochs")
    plt.ylabel("F-beta")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 결과 저장 폴더 생성 및 저장
    Path(f"results/{model_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"results/{model_name}/learning_curve_{dataset_name}.png"
    plt.savefig(save_path)

    print(f"\nLearning curves saved to: {save_path}")

    plt.close()


##################################### #
# validation 결과에 대한 콘퓨전 매트릭스
##################################### #
def visualize_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    dataset_name: str,
    model_name: str,
) -> None:
    """Confusion Matrix를 생성하고 이미지로 저장합니다."""
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(y_true=all_labels, y_pred=all_predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Confusion Matrix ({dataset_name} / {model_name})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 셀에 수치 표시
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    Path(f"results/{model_name}").mkdir(parents=True, exist_ok=True)
    save_path = f"results/{model_name}/confusion_matrix_{dataset_name}.png"
    fig.savefig(save_path)

    print(f"Confusion matrix saved to: {save_path}")
    plt.close(fig)
