import json
import random
import shutil
from pathlib import Path
from typing import Any
from collections import Counter

# 데이터 경로 상수 정의
CLS_DATA_DIR = Path("data/Iron-Scraps/set_with_testset")
CLUSTERED_GROUPS_DIR = CLS_DATA_DIR / "base_clustered_groups_0pct"
ANNOTATION_PATH = CLS_DATA_DIR / "corrected_crop_annotations_0pct.json"

# 출력 폴더 그룹 디렉토리
BASE_SPLIT_DIR = CLS_DATA_DIR / "base_split_folders_0pct"
TRAIN_FOLDERS_DIR = BASE_SPLIT_DIR / "train"
VAL_FOLDERS_DIR = BASE_SPLIT_DIR / "val"

# 분할 비율
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0

def load_annotations(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_folder_info(folder: Path, crops_info: dict) -> tuple[str, int]:
    """폴더 내의 주요 카테고리와 이미지 개수를 반환"""
    images = list(folder.glob("*.jpg"))
    if not images:
        return "unknown", 0

    labels = []
    for img in images:
        meta = crops_info.get(img.name, {})
        cat = meta.get("category_name", "unknown")
        labels.append(cat)

    # 다수결 라벨 (단, 클러스터링이 매우 타이트하게 되었으므로 대체로 동일할 것임)
    counter = Counter(labels)
    majority_label = counter.most_common(1)[0][0]

    return majority_label, len(images)

def greedy_split_folders(category_folders: list[tuple[Path, int]]) -> dict[str, list[Path]]:
    """탐욕 알고리즘으로 폴더를 8:1:1 비율에 가깝게 분배"""
    # 폴더 크기 내림차순 정렬 (큰 덩어리부터 배정해야 오차가 적음)
    category_folders.sort(key=lambda x: x[1], reverse=True)

    total_images = sum(count for _, count in category_folders)

    target_counts = {
        "train": total_images * TRAIN_RATIO,
        "val": total_images * VAL_RATIO,

    }

    current_counts = {"train": 0, "val": 0}
    assigned_folders = {"train": [], "val": []}

    for folder_path, count in category_folders:
        # 현재 목표치 대비 가장 덜 채워진 핀을 찾음 (부족한 비율이 가장 큰 곳)
        best_bin = None
        max_deficit_ratio = -float('inf')

        for bin_name in ["train", "val"]:
            if target_counts[bin_name] == 0:
                continue

            # 현재 배정된 비율
            current_ratio = current_counts[bin_name] / target_counts[bin_name]
            deficit = 1.0 - current_ratio

            if deficit > max_deficit_ratio:
                max_deficit_ratio = deficit
                best_bin = bin_name

        # 만약 전부 목표치를 초과했다면 (거의 없지만), 절대 부족량이 제일 큰 곳으로
        if best_bin is None:
            best_bin = min(current_counts.keys(), key=lambda k: current_counts[k] - target_counts[k])

        assigned_folders[best_bin].append(folder_path)
        current_counts[best_bin] += count

    return assigned_folders, current_counts

def main():
    print("클러스터 폴더 그룹 분할 중...")
    random.seed(2026)

    annotations = load_annotations(ANNOTATION_PATH)
    crops_info = annotations.get("crops", {})

    # 출력 디렉토리 초기화
    if BASE_SPLIT_DIR.exists():
        shutil.rmtree(BASE_SPLIT_DIR)
    TRAIN_FOLDERS_DIR.mkdir(parents=True, exist_ok=True)
    VAL_FOLDERS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 모든 폴더를 순회하며 카테고리별로 분류
    category_groups = {} # category -> list of (folder_path, image_count)

    cluster_folders = [d for d in CLUSTERED_GROUPS_DIR.iterdir() if d.is_dir()]

    for folder in cluster_folders:
        cat_name, img_count = get_folder_info(folder, crops_info)
        if img_count == 0:
            continue

        if cat_name not in category_groups:
            category_groups[cat_name] = []
        category_groups[cat_name].append((folder, img_count))

    global_counts = {"train": 0, "val": 0}

    # 2. 카테고리별로 Greedy 분할 수행
    for category, folders in category_groups.items():
        # 랜덤성을 위해 동일 크기의 폴더끼리는 셔플될 수 있게 약간의 난수 부여
        # 하지만 정렬 안정성을 위해 크기 우선
        random.shuffle(folders)

        assigned, counts = greedy_split_folders(folders)

        print(f"카테고리 {category} -> 총 {sum(counts.values()):4d}장 (폴더 {len(folders)}개)")
        print(f"Train: {counts['train']:4d}장, Val: {counts['val']:4d}장, ")

        # 3. 폴더 복사
        for bin_name, folder_list in assigned.items():
            global_counts[bin_name] += counts[bin_name]

            dest_bin_dir = BASE_SPLIT_DIR / bin_name
            for src_folder in folder_list:
                dest_folder = dest_bin_dir / src_folder.name
                shutil.copytree(src_folder, dest_folder)

    print("\n전체 클러스터 폴더 그룹 분할 및 복사 완료")
    print(f"Train : {global_counts['train']:5d}장")
    print(f"Val   : {global_counts['val']:5d}장")
    print(f"총합  : {sum(global_counts.values()):5d}장")
    print(f"저장 경로: {BASE_SPLIT_DIR}")

if __name__ == "__main__":
    main()
