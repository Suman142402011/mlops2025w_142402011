import torch
from tqdm import tqdm
from config import device

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0
    correct = 0
    for images, labels in tqdm(loader, desc="train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        break  # demo only
    acc = correct / total if total > 0 else 0.0
    return {"loss": loss.item() if 'loss' in locals() else 0.0, "acc": acc}
