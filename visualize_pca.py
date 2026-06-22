import argparse
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def load_hyperparams(file_path: Path) -> dict:
    """하이퍼파라미터 로드 함수"""
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)
    return hyperparams

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparams", type=str, default="./hyperparams.yaml", help="Path to hyperparameter file.")
    return parser.parse_args()

def main():
    args = parse_args()
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(file_path=hyperparam_path)

    # 설정 파일로부터 지원 가능한 모델 및 데이터셋 목록을 동적으로 파싱
    model_options = [k for k, v in hyperparams.get("model", {}).items() if isinstance(v, dict) and "PATH" in v]
    dataset_options = [k for k, v in hyperparams.get("data", {}).items() if isinstance(v, dict) and "DATASET_DIR" in v]

    # CLI args로 넘어오지 않았을 경우 input
    model_name = input(f"Enter model name ({', '.join(model_options)}): ").lower().strip()
    dataset_name = input(f"Enter dataset name ({', '.join(dataset_options)}): ").lower().strip()

    model_config = hyperparams["model"][model_name]
    model_path = Path(model_config["PATH"])
    hub_model_name = model_config["NAME"]

    dataset_config = hyperparams["data"][dataset_name]
    dataset_dir = Path(dataset_config["DATASET_DIR"])
    image_size = hyperparams["data"]["IMAGE_SIZE"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[*] Loading DINOv3 backbone: {hub_model_name}...")
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=hub_model_name,
        source="local",
        weights=str(model_path),
    ).to(device)
    model.eval()

    # 패치 사이즈 추론
    patch_size = model.patch_size if hasattr(model, "patch_size") else 14
    if "16" in hub_model_name: 
        patch_size = 16

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    test_dir = dataset_dir / "test"
    dataset = ImageFolder(root=str(test_dir), transform=transform)

    # 무작위로 5장의 이미지 선택
    num_samples = 5
    indices = random.sample(range(len(dataset)), num_samples)

    images = []
    original_images = []

    for idx in indices:
        img_tensor, _ = dataset[idx]
        images.append(img_tensor)
        # Plotting을 위해 원본 이미지 보존
        img_path = dataset.samples[idx][0]
        original_images.append(Image.open(img_path).convert("RGB").resize((image_size, image_size)))

    x = torch.stack(images).to(device)

    print(f"[*] Extracting features for {num_samples} images...")
    with torch.no_grad():
        features = model.forward_features(x)

    # forward_features의 반환 타입에 맞게 패치 토큰 추출
    if isinstance(features, dict) and 'x_norm_patchtokens' in features:
        patch_tokens = features['x_norm_patchtokens']
    elif isinstance(features, tuple):
        patch_tokens = features[0]
    else:
        patch_tokens = features

    # CLS 토큰이 포함되어 있다면 제외 (보통 N = (H/P) * (W/P) + 1 인 경우)
    H_patch = image_size // patch_size
    W_patch = image_size // patch_size
    expected_patches = H_patch * W_patch
    
    if patch_tokens.shape[1] == expected_patches + 1:
        patch_tokens = patch_tokens[:, 1:, :] # CLS 제외

    B, N, D = patch_tokens.shape
    
    # PCA를 위해 (B*N, D) 2D 행렬로 변환
    features_flat = patch_tokens.reshape(-1, D).cpu().numpy()

    print(f"[*] Applying PCA (reducing from {D} to 3 dimensions)...")
    pca = PCA(n_components=3)
    pca.fit(features_flat)
    pca_features = pca.transform(features_flat)

    # PCA 결과를 RGB 시각화를 위해 [0, 1] 범위로 Min-Max 스케일링
    for i in range(3):
        comp = pca_features[:, i]
        pca_features[:, i] = (comp - comp.min()) / (comp.max() - comp.min())

    # 원래 이미지의 그리드 형태로 재배열 (B, H_patch, W_patch, 3)
    pca_features = pca_features.reshape(B, H_patch, W_patch, 3)

    # 시각화 및 저장
    results_dir = Path(f"results_pca_visualization/{dataset_name}_{hub_model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    for i in range(num_samples):
        # 1행: 원본 이미지
        axes[0, i].imshow(original_images[i])
        axes[0, i].axis('off')
        if i == num_samples // 2:
            axes[0, i].set_title("Original Images", fontsize=20, pad=20)

        # 2행: PCA 컬러맵 이미지
        pca_img = pca_features[i]
        # 부드러운 컬러맵을 위해 (H_patch, W_patch) 크기를 (image_size, image_size)로 보간(Interpolation)
        pca_img_tensor = torch.tensor(pca_img).permute(2, 0, 1).unsqueeze(0) # (1, 3, H_p, W_p)
        pca_img_up = F.interpolate(pca_img_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        pca_img_up = pca_img_up.squeeze(0).permute(1, 2, 0).numpy()

        axes[1, i].imshow(pca_img_up)
        axes[1, i].axis('off')
        if i == num_samples // 2:
            axes[1, i].set_title("PCA Feature Map (Semantic Consistency)", fontsize=20, pad=20)

    plt.tight_layout()
    save_path = results_dir / "pca_visualization.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[*] 🎯 Success! PCA visualization successfully saved to:\n -> {save_path}")

if __name__ == "__main__":
    main()
