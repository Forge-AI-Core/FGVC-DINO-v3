import shutil
from pathlib import Path
from PIL import Image
import json
from collections import Counter

CLS_DATA_DIR = Path("data/Iron-Scraps/set_with_testset")
BASE_TRAIN_FOLDERS = CLS_DATA_DIR / "base_split_folders_0pct/train"
BASE_VAL_FOLDERS = CLS_DATA_DIR / "base_split_folders_0pct/val"

ANNOTATION_PATH = CLS_DATA_DIR / "corrected_crop_annotations_0pct.json"

VANILLA_TRAIN_DIR = CLS_DATA_DIR / "vanilla/split_dataset_0pct/train"
VANILLA_VAL_DIR = CLS_DATA_DIR / "vanilla/split_dataset_0pct/val"


def get_majority_label(folder: Path, crops_info: dict) -> str | None:
    image_paths = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}])
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
        
    return majority_label


def build_vanilla_split(input_dir: Path, output_dir: Path, crops_info: dict, split_name: str) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"{split_name} 입력 폴더가 존재하지 않습니다: {input_dir}")
        return

    cluster_folders = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    total_images_copied = 0
    for folder in cluster_folders:
        majority_label = get_majority_label(folder, crops_info)
        if majority_label is None:
            continue

        dest_cat_dir = output_dir / majority_label
        dest_cat_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}])
        for img_path in image_paths:
            shutil.copy(src=img_path, dst=dest_cat_dir / img_path.name)
            total_images_copied += 1

    print(f"[{split_name}] Vanilla 데이터 구축 완료: 총 {len(cluster_folders)}개 폴더에서 {total_images_copied}장 복사됨.")


def main() -> None:
    print(f"Vanilla Train/Val 올바른 분할 진행 중 (0pct)...")
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    crops_info = annotations.get("crops", {})
    
    build_vanilla_split(BASE_TRAIN_FOLDERS, VANILLA_TRAIN_DIR, crops_info, "Train")
    build_vanilla_split(BASE_VAL_FOLDERS, VANILLA_VAL_DIR, crops_info, "Val")


if __name__ == "__main__":
    main()
