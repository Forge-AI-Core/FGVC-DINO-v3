from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torchvision.transforms import transforms, functional

import argparse
import yaml

import shutil
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

from torch.utils.data import DataLoader
import numpy as np

from typing import Any


# 데이터 루트
CLS_DATA_DIR: Path = Path("data/Iron-Scraps/my_classification")


# 원본 데이터 디렉토리
ORIGINAL_DATA_DIR: Path = CLS_DATA_DIR / "originals"
# 원본 데이터 어노테이션 경로
ORIGINAL_ANNOTATION_PATH: Path = CLS_DATA_DIR / "annotations.json"


# 크롭 데이터 디렉토리
CROPPED_DATA_DIR: Path = CLS_DATA_DIR / "crop_images"
# 크롭 데이터 어노테이션 경로
CROPPED_ANNOTATION_PATH: Path = CLS_DATA_DIR / "crop_annotations.json"


# 임베딩 파일 정보 보관소 디렉토리
EMBEDDING_FILE_DIR: Path = CLS_DATA_DIR / "embeddings"
# 임베딩 파일
EMBEDDING_FILE_PATH: Path = EMBEDDING_FILE_DIR / "embeddings_2nd.npy"
# 이미지 경로 파일
IMAGEPATH_FILE_PATH: Path = EMBEDDING_FILE_DIR / "image_paths_2nd.txt"


# 클러스터링 파일 보관소 디렉토리
CLUSTERED_GROUPS_DIR: Path = CLS_DATA_DIR / "clustered_groups"


# 1차 샘플링 데이터 디렉토리
SAMPLED_OUTPUT_DIR: Path = CLS_DATA_DIR / "sampled_images"


# 2차 클러스터링 파일 보관소 디렉토리
CLUSTERED_GROUPS_DIR_2ND: Path = CLS_DATA_DIR / "clustered_groups_2nd"


# 데이터로더 배치 사이즈
BATCH_SIZE: int = 256


def get_dataloader(batch_size: int) -> tuple[DataLoader, list[Path]]:
    class SquarePad:
        """이미지 비율 유지하며 패딩"""

        def __call__(self, image: Image.Image) -> Image.Image:
            w, h = image.size
            max_length = max(w, h)

            left_pad = (max_length - w) // 2
            top_pad = (max_length - h) // 2
            right_pad = max_length - w - left_pad
            bottom_pad = max_length - h - top_pad

            padding = [
                left_pad,
                top_pad,
                right_pad,
                bottom_pad,
            ]

            return functional.pad(img=image, padding=padding, fill=0)

    image_transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    class CroppedDataset:

        def __init__(self, image_path: list[Path]) -> None:
            self.image_path = image_path

        def __len__(self) -> int:

            return len(self.image_path)

        def __getitem__(self, index: int) -> torch.Tensor:
            image = Image.open(fp=self.image_path[index]).convert("RGB")

            return image_transform(image)

    image_paths = sorted(list(SAMPLED_OUTPUT_DIR.glob(pattern="*.jpg")))

    dataset = CroppedDataset(image_path=image_paths)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return dataloader, image_paths


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="하이퍼 파라미터 로드")
    parser.add_argument(
        "--hyperparams", default="hyperparams.yaml", help="하이퍼 파라미터 경로 입력"
    )
    args = parser.parse_args()

    return args


def load_hyperparams(file_path: Path) -> dict[str, Any]:
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        hyperparams = yaml.safe_load(f)

        return hyperparams


def get_model(model_name: str, model_path: Path) -> nn.Module:
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=model_name,
        source="local",
        weights=str(model_path),
    )

    return model


def get_embeddings(
    dataloader: DataLoader, model: nn.Module, image_paths: list[Path]
) -> torch.Tensor:
    normalized_embeddings = []
    for index, batch in enumerate(dataloader):
        images = batch.to(device)

        with torch.inference_mode():
            features = model(images)

            norm = torch.norm(input=features, p=2, dim=1, keepdim=True)
            normalized_features = features / torch.clamp(input=norm, min=1e-9)
            normalized_embeddings.append(normalized_features.cpu().numpy())

    final_embeddings = np.vstack(normalized_embeddings)

    EMBEDDING_FILE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(file=EMBEDDING_FILE_PATH, arr=final_embeddings)

    IMAGEPATH_FILE_PATH.write_text("\n".join(str(path) for path in image_paths))

    print("임베딩 추출 및 저장 완료")

    return final_embeddings


def apply_clustering(final_embeddings: torch.Tensor, image_paths: list[Path]) -> None:
    # linkage="average" (평균 거리 기준) 또는 "complete" (최대 거리 기준) 사용
    # distance_threshold: 코사인 거리 기준 임계치 (0.15 ~ 0.22 사이 조율)
    # n_clusters=None 설정 시 distance_threshold를 기준으로 군집을 분할합니다.
    clustering = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average", distance_threshold=0.16
    )
    labels = clustering.fit_predict(final_embeddings)

    if CLUSTERED_GROUPS_DIR_2ND.exists():
        shutil.rmtree(CLUSTERED_GROUPS_DIR_2ND)
    CLUSTERED_GROUPS_DIR_2ND.mkdir(parents=True)

    print("클러스터 그룹 폴더 초기화 완료")

    unique_labels = set(labels) - {-1}

    print(
        f"클러스터 폴더 개수 {len(unique_labels)}, 싱글톤 폴더의 개체 {np.sum(labels == -1)}"
    )

    for path, label in zip(image_paths, labels):
        target_folder = CLUSTERED_GROUPS_DIR_2ND / (
            "singletones" if label == -1 else f"cluster_{label:03d}"
        )
        target_folder.mkdir(parents=True, exist_ok=True)

        shutil.copy(src=path, dst=(target_folder / path.name))


if __name__ == "__main__":

    args = parse_cli_args()
    hyperparam_path = Path(args.hyperparams)
    hyperparams = load_hyperparams(file_path=hyperparam_path)

    model_name = hyperparams["model"]["vitb16"]["NAME"]  # 하드코딩
    model_path = hyperparams["model"]["vitb16"]["PATH"]  # 하드코딩

    # model_name = hyperparams["model"]["vit7b16"]["NAME"] # 하드코딩
    # model_path = hyperparams["model"]["vit7b16"]["PATH"] # 하드코딩

    dataloader, image_paths = get_dataloader(batch_size=BATCH_SIZE)

    model = get_model(
        model_name=model_name,
        model_path=model_path,
    )
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    embeddings = get_embeddings(
        dataloader=dataloader,
        model=model,
        image_paths=image_paths,
    )

    apply_clustering(
        final_embeddings=embeddings,
        image_paths=image_paths,
    )
