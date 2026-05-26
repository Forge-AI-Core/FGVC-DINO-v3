import shutil
from pathlib import Path
from PIL import Image


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


# 2차 클러스터링 파일 보관소 디렉토리
CLUSTERED_GROUPS_DIR_2ND = CLS_DATA_DIR / "clustered_groups_2nd"


# 2차 샘플링 데이터 디렉토리
SAMPLED_OUTPUT_DIR_2ND: Path = CLS_DATA_DIR / "sampled_images_2nd"


def get_largest_image(folder: Path) -> Path | None:
    image_paths: list[Path] = sorted(
        [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg"}]
    )

    return max(
        image_paths,
        key=lambda p: Image.open(p).size[0] * Image.open(p).size[1],
        default=None,
    )


def sample_representative_images() -> None:
    if SAMPLED_OUTPUT_DIR_2ND.exists():
        shutil.rmtree(SAMPLED_OUTPUT_DIR_2ND)
    SAMPLED_OUTPUT_DIR_2ND.mkdir(parents=True, exist_ok=True)

    cluster_folders: list[Path] = sorted(
        [d for d in CLUSTERED_GROUPS_DIR_2ND.iterdir() if d.is_dir()]
    )

    success_count: int = 0
    for folder in cluster_folders:
        largest_img = get_largest_image(folder)
        if largest_img is None:
            continue

        destination_path = SAMPLED_OUTPUT_DIR_2ND / largest_img.name
        shutil.copy(src=largest_img, dst=destination_path)

        success_count += 1

    print(
        f"샘플링 완료: 총 {len(cluster_folders)}개 폴더 중 {success_count}개 이미지 샘플링됨."
    )
    print(f"출력 경로: {SAMPLED_OUTPUT_DIR_2ND}")


if __name__ == "__main__":
    sample_representative_images()
