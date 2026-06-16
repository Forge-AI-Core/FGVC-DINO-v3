import shutil
from pathlib import Path
from PIL import Image
import json
from collections import Counter

CLS_DATA_DIR = Path("data/Iron-Scraps/my_classification")
TEST_FOLDERS_DIR = CLS_DATA_DIR / "base_split_folders_10pct/test"
ANNOTATION_PATH = CLS_DATA_DIR / "corrected_crop_annotations_10pct.json"

UNIQUE_TEST_DIR = CLS_DATA_DIR / "unique_sampling/split_dataset_10pct/test"
VANILLA_TEST_DIR = CLS_DATA_DIR / "vanilla/split_dataset_10pct/test"


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


def sample_representative_images() -> None:
    if UNIQUE_TEST_DIR.exists():
        shutil.rmtree(UNIQUE_TEST_DIR)
    if VANILLA_TEST_DIR.exists():
        shutil.rmtree(VANILLA_TEST_DIR)
        
    UNIQUE_TEST_DIR.mkdir(parents=True, exist_ok=True)
    VANILLA_TEST_DIR.mkdir(parents=True, exist_ok=True)

    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    crops_info = annotations.get("crops", {})

    cluster_folders = sorted([d for d in TEST_FOLDERS_DIR.iterdir() if d.is_dir()])

    success_count = 0
    for folder in cluster_folders:
        largest_img = get_filtered_largest_image(folder, crops_info)
        
        if largest_img is None:
            continue

        # Get final category name to place in correct subfolder
        cat_name = crops_info.get(largest_img.name, {}).get("category_name", "unknown")
        
        u_dest = UNIQUE_TEST_DIR / cat_name
        v_dest = VANILLA_TEST_DIR / cat_name
        u_dest.mkdir(parents=True, exist_ok=True)
        v_dest.mkdir(parents=True, exist_ok=True)

        shutil.copy(src=largest_img, dst=u_dest / largest_img.name)
        shutil.copy(src=largest_img, dst=v_dest / largest_img.name)

        success_count += 1

    print(f"Test 샘플링 완료: 총 {len(cluster_folders)}개 폴더 중 {success_count}개 이미지 샘플링됨.")


def main() -> None:
    print("Test 데이터셋 공통 다이어트 진행 중...")
    sample_representative_images()


if __name__ == "__main__":
    main()
