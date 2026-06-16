import json
import shutil
import gc
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, functional
from PIL import Image
import yaml
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm


# 상수 정의
CLS_DATA_DIR = Path("data/Iron-Scraps/set_with_testset")
CROPPED_DATA_DIR = CLS_DATA_DIR / "crop_images_0pct"
ORIGINAL_ANNOTATION_PATH = CLS_DATA_DIR / "crop_annotations_0pct.json"
CORRECTED_ANNOTATION_PATH = CLS_DATA_DIR / "corrected_crop_annotations_0pct.json"
CLUSTERED_GROUPS_DIR = CLS_DATA_DIR / "base_clustered_groups_0pct"
HYPERPARAMS_PATH = Path("hyperparams.yaml")

BATCH_SIZE = 256
CASCADE_STEPS = [
    ("vitl16", 0.12),
    ("vitb16", 0.14),
    ("vits16", 0.16),
]


class SquarePad:
    """이미지 비율 유지하며 패딩"""

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        max_length = max(w, h)
        left_pad = (max_length - w) // 2
        top_pad = (max_length - h) // 2
        right_pad = max_length - w - left_pad
        bottom_pad = max_length - h - top_pad
        padding = [left_pad, top_pad, right_pad, bottom_pad]
        return functional.pad(img=image, padding=padding, fill=0)


class CroppedDataset:
    def __init__(self, image_paths: list[Path]) -> None:
        self.image_paths = image_paths
        self.transform = transforms.Compose(
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

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(fp=self.image_paths[index]).convert("RGB")
        return self.transform(image)


def get_dataloader() -> tuple[DataLoader, list[Path]]:
    image_paths = sorted(list(CROPPED_DATA_DIR.glob("*.jpg")))
    dataset = CroppedDataset(image_paths=image_paths)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )
    return dataloader, image_paths


def load_hyperparams(file_path: Path) -> tuple[dict[str, Any], dict[int, list[str]]]:
    with open(file=file_path, mode="r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model(model_name: str, model_path: str) -> nn.Module:
    print(f"[{model_name}] 모델 로드 중... ({model_path})")
    model = torch.hub.load(
        repo_or_dir="dinov3",
        model=model_name,
        source="local",
        weights=str(model_path),
    )
    return model


def get_embeddings(
    dataloader: DataLoader, model: nn.Module, device: torch.device
) -> np.ndarray:
    print(f"임베딩 추출 시작...")
    normalized_embeddings = []

    for batch in tqdm(dataloader, desc="Extracting"):
        images = batch.to(device)
        with torch.inference_mode():
            features = model(images)
            norm = torch.norm(input=features, p=2, dim=1, keepdim=True)
            normalized_features = features / torch.clamp(input=norm, min=1e-9)
            normalized_embeddings.append(normalized_features.cpu().numpy())

    final_embeddings = np.vstack(normalized_embeddings)
    return final_embeddings


def load_annotations(path: Path) -> tuple[dict[str, Any], dict[int, list[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_category_map(
    annotations: dict[str, Any],
) -> tuple[dict[str, int], dict[int, str]]:
    name_to_id = {}
    id_to_name = {}
    for cat in annotations.get("categories", []):
        name_to_id[cat["name"]] = cat["id"]
        id_to_name[cat["id"]] = cat["name"]
    return name_to_id, id_to_name


def apply_majority_voting(
    embeddings: np.ndarray,
    image_paths: list[Path],
    threshold: float,
    annotations: dict[str, Any],
    model_name: str,
) -> tuple[dict[str, Any], dict[int, list[str]]]:
    print(f"[{model_name}] 클러스터링 시작 (Threshold: {threshold})")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    labels = clustering.fit_predict(embeddings)

    unique_labels = set(labels) - {-1}
    print(
        f"[{model_name}] 군집 폴더 개수: {len(unique_labels)}, 싱글톤 개체: {np.sum(labels == -1)}"
    )

    # 군집별로 파일 그룹화
    clusters = {label: [] for label in unique_labels}
    for path, label in zip(image_paths, labels):
        if label != -1:
            clusters[label].append(path.name)

    crops = annotations.get("crops", {})
    name_to_id, _ = get_category_map(annotations)

    collision_count = 0
    total_corrected_images = 0

    for label, filenames in clusters.items():
        # 각 파일의 현재 라벨(category_name) 수집
        current_labels = []
        for fname in filenames:
            cat_name = crops.get(fname, {}).get("category_name")
            if cat_name:
                current_labels.append(cat_name)

        if not current_labels:
            continue

        # 충돌 확인 (단일 라벨이 아닌 경우)
        unique_cats = set(current_labels)
        if len(unique_cats) > 1:
            collision_count += 1

            # 다수결 판정
            counter = Counter(current_labels)
            max_freq = max(counter.values())
            majority_candidates = [lbl for lbl, count in counter.items() if count == max_freq]

            if len(majority_candidates) > 1:
                # 동률 발생! 해당 클러스터의 모든 파일 라벨 정보를 완전히 삭제 (영구 결번)
                for fname in filenames:
                    if fname in crops:
                        del crops[fname]
            else:
                majority_label = majority_candidates[0]
                majority_id = name_to_id[majority_label]

                # 클러스터 내 모든 파일에 다수결 라벨 덮어쓰기
                for fname in filenames:
                    if fname in crops and crops[fname].get("category_name") != majority_label:
                        crops[fname]["category_name"] = majority_label
                        crops[fname]["category_id"] = majority_id
                        total_corrected_images += 1

    print(
        f"[{model_name}] 🚨 충돌 일어난 클러스터 군집 수: {collision_count}건 해결됨 (총 {total_corrected_images}장 라벨 교정)"
    )

    return annotations, clusters


def main() -> None:
    print("🚀 Cascade Consensus Labeling 시작...")

    # 1. 원본 어노테이션 로드
    annotations = load_annotations(ORIGINAL_ANNOTATION_PATH)

    # 2. 데이터로더 및 하이퍼파라미터 로드
    dataloader, image_paths = get_dataloader()
    hyperparams = load_hyperparams(HYPERPARAMS_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Cascade Steps 순회
    for model_name, threshold in CASCADE_STEPS:
        print(f"\n{'='*50}")
        print(f"▶ Step: {model_name} (Threshold: {threshold})")
        print(f"{'='*50}")

        # 모델 로드
        actual_model_name = hyperparams["model"][model_name]["NAME"]
        model_path = hyperparams["model"][model_name]["PATH"]
        model = get_model(actual_model_name, model_path)
        model.to(device)
        model.eval()

        # 임베딩 추출
        embeddings = get_embeddings(dataloader, model, device)

        # 모델 메모리 강제 해제 (OOM 방지)
        print(f"[{model_name}] 메모리 해제 중...")
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # 클러스터링 및 다수결 교정 (메모리 상의 annotations 계속 업데이트)
        annotations, last_clusters = apply_majority_voting(
            embeddings=embeddings,
            image_paths=image_paths,
            threshold=threshold,
            annotations=annotations,
            model_name=model_name,
        )

    # 4. 최종 결과 저장
    print("\n✅ 모든 체급의 릴레이 교정이 완료되었습니다.")
    save_annotations(annotations, CORRECTED_ANNOTATION_PATH)

    # 5. 최종 군집을 물리적 폴더(base_clustered_groups)로 분리 생성
    print("\n🚀 물리적 클러스터 폴더(base_clustered_groups) 생성 시작...")
    if CLUSTERED_GROUPS_DIR.exists():
        shutil.rmtree(CLUSTERED_GROUPS_DIR)
    CLUSTERED_GROUPS_DIR.mkdir(parents=True)
    
    crops = annotations.get("crops", {})
    for label, filenames in last_clusters.items():
        valid_filenames = [f for f in filenames if f in crops]
        if not valid_filenames:
            continue
            
        target_folder = CLUSTERED_GROUPS_DIR / f"cluster_{label:03d}"
        target_folder.mkdir(parents=True, exist_ok=True)
        for fname in valid_filenames:
            src_path = CROPPED_DATA_DIR / fname
            if src_path.exists():
                shutil.copy(src_path, target_folder / fname)
                
    print(f"🎉 물리적 클러스터 폴더 복사 완료: {CLUSTERED_GROUPS_DIR}")

    print(f"🎉 최종 저장 위치: {CORRECTED_ANNOTATION_PATH}")


if __name__ == "__main__":
    main()
