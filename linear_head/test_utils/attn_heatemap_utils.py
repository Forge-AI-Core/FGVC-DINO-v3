import torch
import torch.nn.functional as F
from torch import nn
from typing import Any
import types
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


######################### #
# 어텐션 맵 캡처용 전역 변수
######################### #
captured_attn_weights: torch.Tensor | None = None


################################# #
# 마지막 블럭의 어텐션 몽키 패치 함수
################################# #
def patch_attention_module(model: nn.Module) -> None:
    """DINOv3의 마지막 block의 SelfAttention 모듈에 compute_attention 몽키 패치를 적용합니다.
    Args:
        model (nn.Module): LoRA가 적용된 PeftModel 인스턴스
    """
    # lora로 래핑되어있는 경우 (로라)model.base_model의 .model 어트리뷰트를 가져와야 합니다.
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    # 마지막 블록의 어텐션 객체 생성
    last_attn = base_vit.blocks[-1].attn

    ############################################################### #
    # DINOv3의 SelfAttention클래스에 존재하는 compute_attention 메서드
    ############################################################### #
    def custom_compute_attention(
        self: Any, qkv: torch.Tensor, attn_bias: Any = None, rope: Any = None
    ) -> torch.Tensor:
        """어텐션 연산 시 attention map을 가로채기 위한 커스텀 함수입니다.
        Args:
            self (Any): SelfAttention 인스턴스
            qkv (torch.Tensor): QKV 융합 텐서
            attn_bias (Any, optional): 어텐션 바이어스
            rope (Any, optional): Rotary Position Embedding 정보
        Returns:
            torch.Tensor: 어텐션 연산 결과 텐서
        """
        global captured_attn_weights
        assert attn_bias is None

        # 입력 텐서의 형상 확인
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        # Q,K,V 분리
        qkv_reshaped = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv_reshaped, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # Attention weights 직접 계산 [B, heads, N, N]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # CPU 메모리로 전송하여 캡처 저장
        captured_attn_weights = attn_weights.detach().cpu()

        # 실제 어텐션 출력을 얻기 위해 scaled_dot_product_attention 수행
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)

        return x.reshape([B, N, C])

    last_attn.compute_attention = types.MethodType(custom_compute_attention, last_attn)


####################################################### #
# 마지막 블록의 어텐션을 트래킹한 결과물로 히트맵을 그리는 함수
####################################################### #
def generate_attention_heatmap(
    model: nn.Module,
    img_tensor: torch.Tensor,
    image_size: int,
    patch_size: int,
) -> np.ndarray:
    """모델 포워드 패스를 실행하고, 캡처된 어텐션 가중치로부터 2D 히트맵을 생성합니다.
    Args:
        model (nn.Module): LoRA 모델
        img_tensor (torch.Tensor): 모델 입력 텐서 [1, 3, H, W]
        image_size (int): 입력 이미지 크기
        patch_size (int): 패치 크기
    Returns:
        np.ndarray: [image_size, image_size] 크기의 2D 히트맵 배열 (0 ~ 1 정규화)
    """
    global captured_attn_weights

    attn = captured_attn_weights[0]  # [heads, N, N]
    base_vit = model.base_model.model if hasattr(model, "base_model") else model
    n_storage_tokens = base_vit.n_storage_tokens

    cls_attn = attn[:, 0, n_storage_tokens + 1 :]
    cls_attn_mean = cls_attn.mean(dim=0)

    grid_size = image_size // patch_size
    cls_attn_grid = cls_attn_mean.reshape(1, 1, grid_size, grid_size)

    attn_resized = F.interpolate(
        cls_attn_grid,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    heatmap = attn_resized.squeeze().numpy()
    h_min, h_max = heatmap.min(), heatmap.max()
    heatmap_normalized = (heatmap - h_min) / (h_max - h_min + 1e-8)

    return heatmap_normalized


################## #
# 어텐션 히트맵 플롯
################## #
def save_visualization_plot(
    orig_img: Image.Image,
    heatmap: np.ndarray,
    true_label_name: str,
    pred_label_name: str,
    confidence: float,
    save_path: Path,
) -> None:
    """원본 이미지, 어텐션 오버레이, 순수 히트맵을 가로로 결합하여 이미지 파일로 저장합니다.
    Args:
        orig_img (PIL.Image.Image): SquarePad 처리된 원본 PIL 이미지
        heatmap (np.ndarray): 2D 정규화 히트맵
        true_label_name (str): 실제 클래스 이름
        pred_label_name (str): 예측 클래스 이름
        confidence (float): 예측 클래스 확률
        save_path (Path): 결과 저장 경로
    """
    h, w = heatmap.shape
    orig_img_resized = orig_img.resize((w, h), resample=Image.Resampling.BILINEAR)
    orig_np = np.array(orig_img_resized) / 255.0

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # Panel 1: Original Image
    axes[0].imshow(orig_np)
    axes[0].set_title(
        f"Original (True: {true_label_name})", fontsize=12, fontweight="bold"
    )
    axes[0].axis("off")

    # Panel 2: Attention Overlay
    axes[1].imshow(orig_np)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title(
        f"Overlay (Pred: {pred_label_name} | {confidence * 100.0:.1f}%)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Panel 3: Pure Heatmap
    im = axes[2].imshow(heatmap, cmap="jet")
    axes[2].set_title("Self-Attention Map", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
