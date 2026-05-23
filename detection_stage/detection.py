from ultralytics import YOLO


def train_detector() -> None:
    """YOLOv11m 모델을 사용해 1클래스 객체 제안기(Object Proposal) 학습을 구동합니다.

    L4 GPU (24GB VRAM) 환경에 맞춰 입력 해상도(imgsz=1920) 및 배치 사이즈(batch=4)를 최적화했습니다.
    """
    model = YOLO("yolo11m.pt")

    model.train(
        data="yolo_data.yaml",  # 설정 파일 경로
        epochs=300,  # 1클래스 수렴 속도를 고려해 200 에포크로 설정
        patience=50,  # 조기 종료 임계값
        imgsz=1920,  # 원거리 소형 객체 보존을 위해 1920 해상도 적용
        batch=8,  # 24GB VRAM OOM 방지를 위한 안전한 배치 사이즈
        device=0,  # GPU 0번 사용
        workers=4,  # 데이터 로더 워커 수
    )


if __name__ == "__main__":
    train_detector()
