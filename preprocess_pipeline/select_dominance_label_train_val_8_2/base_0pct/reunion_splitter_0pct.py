import shutil
import random
from pathlib import Path
import json
from collections import defaultdict

CLS_DATA_DIR = Path("data/Iron-Scraps/set_with_testset")
BASE_TRAIN_FOLDERS = CLS_DATA_DIR / "base_split_folders_0pct/train"
BASE_VAL_FOLDERS = CLS_DATA_DIR / "base_split_folders_0pct/val"

ANNOTATION_PATH = CLS_DATA_DIR / "corrected_crop_annotations_0pct.json"

VANILLA_TRAIN_DIR = CLS_DATA_DIR / "vanilla/split_dataset_0pct/train"
VANILLA_VAL_DIR = CLS_DATA_DIR / "vanilla/split_dataset_0pct/val"


def main() -> None:
    print("Vanilla Train/Val Reunion 및 무작위 분할 진행 중...")
    random.seed(2026)
    
    if VANILLA_TRAIN_DIR.exists():
        shutil.rmtree(VANILLA_TRAIN_DIR)
    if VANILLA_VAL_DIR.exists():
        shutil.rmtree(VANILLA_VAL_DIR)
        
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    crops_info = annotations.get("crops", {})

    # 1. 모든 Train, Val 클러스터 폴더 안의 이미지 수집 (Reunion)
    category_pools = defaultdict(list)
    
    # Train 폴더 수집
    if BASE_TRAIN_FOLDERS.exists():
        for cluster_folder in BASE_TRAIN_FOLDERS.iterdir():
            if not cluster_folder.is_dir(): continue
            for img_path in cluster_folder.glob("*.jpg"):
                cat_name = crops_info.get(img_path.name, {}).get("category_name", "unknown")
                category_pools[cat_name].append(img_path)
                
    # Val 폴더 수집
    if BASE_VAL_FOLDERS.exists():
        for cluster_folder in BASE_VAL_FOLDERS.iterdir():
            if not cluster_folder.is_dir(): continue
            for img_path in cluster_folder.glob("*.jpg"):
                cat_name = crops_info.get(img_path.name, {}).get("category_name", "unknown")
                category_pools[cat_name].append(img_path)

    # 2. 카테고리별로 8:1 비율로 분할하여 Vanilla 폴더에 복사
    # 원래 8:2이므로, Train과 Val의 상대 비율은 8:2 입니다.
    # 즉 Train은 8/10, Val은 2/10 입니다.
    train_ratio = 8.0 / 10.0
    
    total_train = 0
    total_val = 0
    
    for category, images in category_pools.items():
        random.shuffle(images)
        total_count = len(images)
        train_count = int(total_count * train_ratio)
        
        train_images = images[:train_count]
        val_images = images[train_count:]
        
        train_cat_dir = VANILLA_TRAIN_DIR / category
        val_cat_dir = VANILLA_VAL_DIR / category
        train_cat_dir.mkdir(parents=True, exist_ok=True)
        val_cat_dir.mkdir(parents=True, exist_ok=True)
        
        for img in train_images:
            shutil.copy(img, train_cat_dir / img.name)
            
        for img in val_images:
            shutil.copy(img, val_cat_dir / img.name)
            
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"카테고리 {category} -> 총 {total_count:4d}장 | Train: {len(train_images):4d}장, Val: {len(val_images):4d}장")

    print("\nVanilla Reunion 분할 및 복사 완료")
    print(f"Train : {total_train:5d}장")
    print(f"Val   : {total_val:5d}장")
    print(f"총합  : {total_train + total_val:5d}장")


if __name__ == "__main__":
    main()
