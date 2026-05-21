from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader

import PIL


def get_data_loader(
    dataset_dir: Path, batch_size: int = 32, image_size: int = 224
) -> tuple[DataLoader, DataLoader, DataLoader]:

    class SquarePad:
        def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
            w, h = image.size
            max_wh = max(w, h)

            pad_left = int((max_wh - w) / 2)
            pad_top = int((max_wh - h) / 2)
            pad_right = max_wh - w - pad_left
            pad_bottom = max_wh - h - pad_top

            padding = (pad_left, pad_top, pad_right, pad_bottom)

            return F.pad(img=image, padding=padding)

    transform = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 파이토치의 이미지폴더는 경로 폴더 하위 폴더를 바로 클래스명으로 인식합니다.
    # 따라서, 경로를 정확하게 클래스 폴더 상위폴더로 잡아줘야 합니다.
    # 이미지폴더로 생성한 객체는 데이터셋 타입으로 데이터로더에 즉시 인자로 넣을 수 있습니다.
    train_dataset = ImageFolder(root=dataset_dir / "train", transform=transform)
    test_dataset = ImageFolder(root=dataset_dir / "test", transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    shuffled_test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader, shuffled_test_loader
