#!/usr/bin/env python3
"""Bounding Box Cropper for FGVC-DINO-v3 dataset.

This script parses COCO format annotations.json, crops the specified object
bounding boxes from base images, and saves them to the original_crop_images directory.
It also outputs a mapping file tracking the cropped images' original metadata and labels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from PIL import Image
from tqdm import tqdm


# 데이터 루트
CLS_DATA_DIR = Path("data/Iron-Scraps/my_classification")


# 원본 데이터 디렉토리
ORIGINAL_DATA_DIR = CLS_DATA_DIR / "originals"
# 원본 데이터 어노테이션 경로
ORIGINAL_ANNOTATION_PATH = CLS_DATA_DIR / "annotations.json"


# 크롭 데이터 디렉토리
CROPPED_DATA_DIR = CLS_DATA_DIR / "crop_images"
# 크롭 데이터 어노테이션 경로
CROPPED_ANNOTATION_PATH = CLS_DATA_DIR / "crop_annotations.json"


# 임베딩 파일 정보 보관소 디렉토리
EMBEDDING_FILE_DIR = CLS_DATA_DIR / "embeddings"
# 임베딩 파일
EMBEDDING_FILE_PATH = EMBEDDING_FILE_DIR / "embeddings.npy"
# 이미지 경로 파일
IMAGEPATH_FILE_PATH = EMBEDDING_FILE_DIR / "image_paths.txt"


# 클러스터링 파일 보관소 디렉토리
CLUSTERED_GROUPS_DIR = CLS_DATA_DIR / "clustered_groups"


def load_coco_annotations(annotation_path: Path) -> dict[str, Any]:
    """원본 어노테이션 로드"""
    with open(annotation_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    return json_data


def build_image_lookup(images: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """이미지 정보 조회용 딕셔너리 생성"""
    image_lookup = {image["id"]: image for image in images}

    return image_lookup


def build_category_lookup(categories: list[dict[str, Any]]) -> dict[int, str]:
    category_lookup = {category["id"]: category["name"] for category in categories}

    return category_lookup


def compute_crop_box(
    bbox: list[float],
    img_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    img_w, img_h = img_size
    x, y, w, h = bbox

    # 이미지에서 좌표 순서는 항상 좌상단에서 우하단
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(img_w, int(round(x + w)))
    bottom = min(img_h, int(round(y + h)))

    return left, top, right, bottom


def is_valid_box(left: int, top: int, right: int, bottom: int) -> bool:

    return right > left and bottom > top


def crop_and_save_image(
    image_path: Path,
    crop_box: tuple[int, int, int, int],
    save_path: Path,
) -> None:
    with Image.open(image_path) as img:
        cropped = img.crop(crop_box).convert("RGB")
        cropped.save(save_path, "JPEG", quality=100, subsampling=0)


def process_crop(
    annotation: dict[str, Any],
    base_dir: Path,
    output_dir: Path,
    image_lookup: dict[int, dict[str, Any]],
    category_lookup: dict[int, str],
) -> dict[str, Any] | None:
    image_id: int = annotation["image_id"]
    bbox: list[float] = annotation.get("bbox", [])
    annotation_id: int = annotation["id"]
    category_id: int = annotation["category_id"]

    img_info = image_lookup[image_id]
    original_filename = Path(img_info["file_name"]).name
    src_image_path = base_dir / original_filename

    with Image.open(src_image_path) as temp_img:
        img_size = temp_img.size

    left, top, right, bottom = compute_crop_box(bbox, img_size)
    if not is_valid_box(left, top, right, bottom):

        print(f"크롭박스 좌표가 이상합니다. {annotation_id}, {image_id}")
        print(f"bbox: {bbox}, img_size: {img_size}")
        print(f"left: {left}, top: {top}, right: {right}, bottom: {bottom}")

        return None

    src_path_obj = Path(original_filename)
    crop_filename = f"{src_path_obj.stem}_crop_{annotation_id}.jpg"
    destination_image_path = output_dir / crop_filename

    crop_and_save_image(
        src_image_path, (left, top, right, bottom), destination_image_path
    )

    return {
        "filename": crop_filename,
        "metadata": {
            "category_id": category_id,
            "category_name": category_lookup.get(category_id, "unknown"),
            "original_image": original_filename,
            "original_image_id": image_id,
            "annotation_id": annotation_id,
            "bbox": bbox,
        },
    }


def run_pipeline(annotation_path: Path) -> None:
    coco_data = load_coco_annotations(annotation_path)

    image_lookup = build_image_lookup(coco_data.get("images", []))
    category_lookup = build_category_lookup(coco_data.get("categories", []))
    annotations = coco_data.get("annotations", [])

    CROPPED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    crops_metadata: dict[str, Any] = {}
    processed_count = 0
    success_count = 0

    iterator = tqdm(annotations, desc="크롭 작업 중")

    for annotation in iterator:
        result = process_crop(
            annotation,
            ORIGINAL_DATA_DIR,
            CROPPED_DATA_DIR,
            image_lookup,
            category_lookup,
        )
        processed_count += 1

        if result is not None:
            crops_metadata[result["filename"]] = result["metadata"]
            success_count += 1

    crop_anno_payload = {
        "categories": coco_data.get("categories", []),
        "crops": crops_metadata,
    }
    with open(CROPPED_ANNOTATION_PATH, "w", encoding="utf-8") as f:
        json.dump(crop_anno_payload, f, indent=4, ensure_ascii=False)

    print(
        "파이프라인 작동 완료",
        "전체 대상: ",
        processed_count,
        "성공: ",
        success_count,
        "출력 디렉토리: ",
        CROPPED_DATA_DIR,
        "크롭 어노테이션 경로: ",
        CROPPED_ANNOTATION_PATH,
    )


if __name__ == "__main__":
    run_pipeline(annotation_path=ORIGINAL_ANNOTATION_PATH)
