from ultralytics import YOLO


def train_detector():
    model = YOLO("yolo11l.pt")

    results = model.train(
        data="yolo_data.yaml",  # 설정 파일 경로
        epochs=500,  # 총 epoch 수
        patience=100,  # 조기 종료 임계값 (원래 50)
        imgsz=1280,  # 이미지 입력 크기 상향 (원래 640)
        batch=8,  # 배치 사이즈 (L4 24GB VRAM에서 500 epoch 동안 Validation OOM이 절대 나지 않도록 안정적으로 설정)
        device=0,  # GPU 0번 사용 (CPU 사용 시 'cpu')
        workers=4,  # 데이터 로더 멀티프로세스 개수
        cos_lr=True,  # 500 epoch 장기 학습에 맞춰 코사인 러닝레이트 스케줄러 적용
        save_period=10,  # GCP 인스턴스 중단에 대비하여 10 epoch마다 체크포인트 저장
        mosaic=0.0,  # 소형 객체 픽셀 깨짐 방지를 위해 Mosaic 비활성화
        mixup=0.0,  # 형태 보존을 위해 mixup 비활성화
    )


if __name__ == "__main__":
    train_detector()
