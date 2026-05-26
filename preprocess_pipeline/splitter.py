import json
import random
import shutil
from pathlib import Path
from typing import Any


# 데이터 경로 상수 정의
CLS_DATA_DIR: Path = Path("data/Iron-Scraps/my_classification")
SAMPLED_IMAGES_DIR: Path = CLS_DATA_DIR / "sampled_images_2nd"
ANNOTATION_PATH: Path = CLS_DATA_DIR / "crop_annotations.json"
OUTPUT_SPLIT_DIR: Path = CLS_DATA_DIR / "split_dataset"


# 분할 비율 (Train: 80%, Val: 10%, Test: 10%)
TRAIN_RATIO: float = 0.8
VAL_RATIO: float = 0.1
TEST_RATIO: float = 0.1


def load_annotations(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_images_by_category(
    image_dir: Path,
    annotations: dict[str, Any],
) -> dict[str, list[Path]]:
    crops_info: dict[str, Any] = annotations.get("crops", {})
    category_groups: dict[str, list[Path]] = {}

    image_paths: list[Path] = sorted(image_dir.glob("*.jpg"))

    for img_path in image_paths:
        filename: str = img_path.name
        crop_metadata = crops_info.get(filename)
        category_name: str = crop_metadata.get("category_name", "unknown")

        if category_name not in category_groups:
            category_groups[category_name] = []
        category_groups[category_name].append(img_path)

    return category_groups


def split_and_copy_dataset(
    category_groups: dict[str, list[Path]], output_dir: Path
) -> None:
    # 재현성을 위한 셔플 시드 고정
    random.seed(2026)

    # 출력 디렉토리 초기화
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for category, paths in category_groups.items():
        shuffled_paths: list[Path] = paths.copy()
        random.shuffle(shuffled_paths)

        total_count: int = len(shuffled_paths)

        train_end: int = int(total_count * TRAIN_RATIO)
        val_end: int = train_end + int(total_count * VAL_RATIO)

        splits: dict[str, list[Path]] = {
            "train": shuffled_paths[:train_end],
            "val": shuffled_paths[train_end:val_end],
            "test": shuffled_paths[val_end:],
        }

        for split_name, split_paths in splits.items():
            dest_dir: Path = output_dir / split_name / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            for src_path in split_paths:
                dest_path: Path = dest_dir / src_path.name
                shutil.copy(src=src_path, dst=dest_path)
                split_counts[split_name] += 1

        print(
            f"카테고리 {category} -> 총 {total_count:4d}장, \n"
            f"Train: {len(splits['train']):4d}장, Val: {len(splits['val']):4d}장, Test: {len(splits['test']):4d}장"
        )

    print("전체 데이터셋 분할 및 복사 완료")
    print(f"Train : {split_counts['train']:5d}장")
    print(f"Val   : {split_counts['val']:5d}장")
    print(f"Test  : {split_counts['test']:5d}장")
    print(f"총합  : {sum(split_counts.values()):5d}장")
    print(f"저장 경로: {output_dir}")


def main() -> None:

    print("데이터셋 분할 중...")

    annotations = load_annotations(ANNOTATION_PATH)

    category_groups = group_images_by_category(SAMPLED_IMAGES_DIR, annotations)

    split_and_copy_dataset(category_groups, OUTPUT_SPLIT_DIR)


if __name__ == "__main__":
    main()
