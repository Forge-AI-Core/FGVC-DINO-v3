import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """전체 데이터로더를 순회하며 모델의 예측 확률과 참 라벨을 수집합니다.

    Args:
        model (nn.Module): 평가 대상 PyTorch 모델
        data_loader (DataLoader): 평가용 데이터로더
        device (torch.device): 연산을 수행할 디바이스 (CPU 또는 CUDA)

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - y_true: [N] 크기의 정수 라벨 배열
            - y_probs: [N, C] 크기의 클래스별 확률값 배열
    """
    model.eval()
    all_probs = []
    all_targets = []

    with torch.inference_mode():
        for images, targets in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    return y_true, y_probs


def calculate_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
) -> dict[str, any]:
    """각 클래스별 및 Micro-average에 대해 Precision-Recall Curve 데이터와 AP를 계산합니다.

    Args:
        y_true (np.ndarray): 참 라벨 배열 [N]
        y_probs (np.ndarray): 클래스별 확률 배열 [N, C]
        num_classes (int): 전체 클래스 개수

    Returns:
        dict[str, any]: 각 클래스 인덱스 또는 'micro' 키를 가지고,
                        그 아래에 'precision', 'recall', 'ap' 값을 담은 딕셔너리
    """
    results = {}

    # 1. 원핫 인코딩
    y_true_onehot = np.eye(num_classes)[y_true]

    # 2. 클래스별 PR 커브 및 AP 계산
    for class_idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, class_idx],
            y_probs[:, class_idx],
        )
        ap = average_precision_score(
            y_true=y_true_onehot[:, class_idx],
            y_score=y_probs[:, class_idx],
        )
        results[str(class_idx)] = {
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }

    # 3. Micro-average PR 커브 및 AP 계산
    precision_micro, recall_micro, _ = precision_recall_curve(
        y_true_onehot.ravel(),
        y_probs.ravel(),
    )
    ap_micro = average_precision_score(
        y_true=y_true_onehot,
        y_score=y_probs,
        average="micro",
    )
    results["micro"] = {
        "precision": precision_micro,
        "recall": recall_micro,
        "ap": ap_micro,
    }

    return results


def plot_pr_curve(
    pr_data: dict[str, any],
    class_names: list[str],
    save_path: Path,
) -> None:
    """계산된 PR Curve 데이터를 바탕으로 세련된 트레이드오프 곡선을 그려 저장합니다.

    Args:
        pr_data (dict[str, any]): 클래스별 precision, recall, ap가 포함된 데이터
        class_names (list[str]): 클래스 이름 리스트
        save_path (Path): 그래프 이미지를 저장할 경로
    """
    plt.figure(figsize=(9, 8))

    # Premium Harmonious Palette (Modern vibrant colors)
    colors = ["#4361EE", "#4CC9F0", "#F72585", "#7209B7"]

    # Micro-average plot first (dashed line)
    micro = pr_data["micro"]
    plt.plot(
        micro["recall"],
        micro["precision"],
        color=colors[3],
        linestyle="--",
        linewidth=2.5,
        label=f"Micro-average (AP = {micro['ap']:.4f})",
    )

    # Class-wise plots
    for class_idx, class_name in enumerate(class_names):
        class_str = str(class_idx)
        if class_str not in pr_data:
            continue
        data = pr_data[class_str]
        color = colors[class_idx % len(colors)]
        plt.plot(
            data["recall"],
            data["precision"],
            color=color,
            linewidth=2.0,
            label=f"Class '{class_name}' (AP = {data['ap']:.4f})",
        )

    plt.title(
        "Precision-Recall Trade-off Curves",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Recall (Sensitivity/Reactivity)", fontsize=11, labelpad=10)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=10)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    # Premium styling with subtle grid and legend formatting
    plt.grid(visible=True, linestyle=":", alpha=0.6, color="#CCCCCC")
    plt.legend(
        loc="lower left",
        frameon=True,
        facecolor="#F8F9FA",
        edgecolor="#E9ECEF",
        fontsize=10,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_ap_summary(pr_data: dict[str, any], class_names: list[str]) -> None:
    """터미널에 Average Precision(AP) 요약을 구조화된 테이블 형태로 출력합니다.

    Args:
        pr_data (dict[str, any]): PR Curve 및 AP 데이터
        class_names (list[str]): 클래스 이름 리스트
    """
    header = f"{'Class / Target':<25} | {'Average Precision (AP)':<25}"
    separator = "-" * len(header)

    print("\n" + separator)
    print(header)
    print(separator)

    # Class-wise AP
    for class_idx, class_name in enumerate(class_names):
        class_str = str(class_idx)
        if class_str in pr_data:
            ap_val = pr_data[class_str]["ap"]
            print(f"{class_name:<25} | {ap_val:.4f}")

    print(separator)
    # Micro AP
    micro_ap = pr_data["micro"]["ap"]
    print(f"{'Micro-average':<25} | {micro_ap:.4f}")
    print(separator + "\n")


def generate_and_save_pr_curve(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    save_path: Path,
) -> None:
    """전체 테스트 데이터셋을 통해 추론을 수행하여 PR 커브를 계산, 시각화하고 AP 요약을 저장합니다.
    Args:
        model (nn.Module): 평가할 모델
        test_loader (DataLoader): 테스트 데이터셋 로더
        device (torch.device): 가속기 디바이스
        class_names (list[str]): 클래스 명칭 리스트
        save_path (Path): 그래프 이미지 저장 경로
    """
    print("[*] Compiling model predictions on full test dataset for PR curve...")
    y_true, y_probs = collect_predictions(model, test_loader, device)

    num_classes = len(class_names)
    pr_data = calculate_pr_curves(y_true, y_probs, num_classes)

    plot_pr_curve(pr_data, class_names, save_path)

    print(f"[*] Precision-Recall curve plotted and saved to: {save_path.resolve()}")

    print_ap_summary(pr_data, class_names)
