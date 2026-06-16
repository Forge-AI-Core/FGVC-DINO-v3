import shutil
from pathlib import Path
from PIL import Image
import json
from collections import Counter

CLS_DATA_DIR = Path("data/Iron-Scraps/my_classification")
BASE_TRAIN_FOLDERS = CLS_DATA_DIR / "base_split_folders_25pct/train"
BASE_VAL_FOLDERS = CLS_DATA_DIR / "base_split_folders_25pct/val"

ANNOTATION_PATH = CLS_DATA_DIR / "corrected_crop_annotations_25pct.json"

UNIQUE_TRAIN_DIR = CLS_DATA_DIR / "unique_sampling/split_dataset_25pct/train"
UNIQUE_VAL_DIR = CLS_DATA_DIR / "unique_sampling/split_dataset_25pct/val"


def get_filtered_largest_image(folder: Path, crops_info: dict) -> Path | None:
    image_paths: list[Path] = sorted(
        [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}]
    )
    
    if not image_paths:
        return None

    labels = []
    for p in image_paths:
        meta = crops_info.get(p.name, {})
        cat = meta.get("category_name", "unknown")
        labels.append(cat)
        
    counter = Counter(labels)
    max_freq = max(counter.values())
    majority_candidates = [lbl for lbl, count in counter.items() if count == max_freq]
    
    # 동률 시 danger 우선 정책
    if "danger" in majority_candidates:
        majority_label = "danger"
    else:
        majority_label = sorted(majority_candidates)[0]
        
    filtered_paths = [p for p, lbl in zip(image_paths, labels) if lbl == majority_label]

    largest_img = max(
        filtered_paths,
        key=lambda p: Image.open(p).size[0] * Image.open(p).size[1],
        default=None,
    )
    
    return largest_img


def sample_split(input_dir: Path, output_dir: Path, crops_info: dict, split_name: str) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"{split_name} 입력 폴더가 존재하지 않습니다: {input_dir}")
        return

    cluster_folders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    success_count = 0
    for folder in cluster_folders:
        largest_img = get_filtered_largest_image(folder, crops_info)
        
        if largest_img is None:
            continue

        cat_name = crops_info.get(largest_img.name, {}).get("category_name", "unknown")
        dest_cat_dir = output_dir / cat_name
        dest_cat_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(src=largest_img, dst=dest_cat_dir / largest_img.name)
        success_count += 1

    print(f"[{split_name}] 샘플링 완료: 총 {len(cluster_folders)}개 폴더 중 {success_count}장 샘플링됨.")


def main() -> None:
    print("Unique Train/Val 데이터 다이어트 진행 중...")
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    crops_info = annotations.get("crops", {})
    
    sample_split(BASE_TRAIN_FOLDERS, UNIQUE_TRAIN_DIR, crops_info, "Train")
    sample_split(BASE_VAL_FOLDERS, UNIQUE_VAL_DIR, crops_info, "Val")


if __name__ == "__main__":
    main()
