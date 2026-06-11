import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


########## #
# 학습 함수
########## #
def train_model(
    device: torch.device,
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    num_epochs: int = 1,
) -> tuple[float, float]:
    model.train()

    # initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    avg_loss = 0.0
    train_acc = 0.0

    # learning loop
    progress_bar = tqdm(
        iterable=train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"
    )
    for index, batch in enumerate(progress_bar):
        optimizer.zero_grad()

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        # loss
        running_loss += loss.item()
        avg_loss = running_loss / (index + 1)

        # accuracy
        predictions = logits.argmax(dim=1)
        correct += predictions.eq(labels).sum().item()
        total += labels.size(0)
        train_acc = 100.0 * correct / total if total > 0 else 0.0

        # postfix
        progress_bar.set_postfix(
            {"avg_loss": f"{avg_loss:.4f}", "train_acc": f"{train_acc:.2f}%"}
        )

    print(
        f"epoch {epoch+1}/{num_epochs}, train loss: {avg_loss:.4f}, train acc: {train_acc:.2f}%"
    )

    return avg_loss, train_acc
