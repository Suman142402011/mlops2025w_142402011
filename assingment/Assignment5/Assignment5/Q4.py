import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import wandb
from torch.utils.data import Subset

# -----------------------------
# WandB setup
# -----------------------------

wandb.init(project="cifar-transfer-learning", name="CIFAR100→CIFAR10-then-reverse")

# -----------------------------
# Config
# -----------------------------
config = {
    "epochs": 20,
    "batch_size": 64,
    "lr": 1e-3,
    "subset_ratio": 0.1,
}
wandb.config.update(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Data preparation
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def get_subset(dataset, ratio):
    subset_size = int(len(dataset) * ratio)
    indices = torch.randperm(len(dataset))[:subset_size]
    return Subset(dataset, indices)

# CIFAR-10
train10 = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test10 = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# CIFAR-100
train100 = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
test100 = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

train10_small = get_subset(train10, config["subset_ratio"])
train100_small = get_subset(train100, config["subset_ratio"])

# -----------------------------
# Model
# -----------------------------
model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 100)  # For CIFAR-100 initially
model = model.to(device)

criterion = nn.CrossEntropyLoss()

def train(model, loader, optimizer, epochs, dataset_name):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        wandb.log({f"{dataset_name}_loss": running_loss / len(loader),
                   f"{dataset_name}_accuracy": acc,
                   "epoch": epoch + 1})
        print(f"[{dataset_name}] Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(loader):.3f} | Acc: {acc:.2f}%")

# -----------------------------
# Training: CIFAR-100 → CIFAR-10
# -----------------------------
trainloader100 = torch.utils.data.DataLoader(train100_small, batch_size=config["batch_size"], shuffle=True)
trainloader10 = torch.utils.data.DataLoader(train10_small, batch_size=config["batch_size"], shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("\nTraining on CIFAR-100...")
train(model, trainloader100, optimizer, config["epochs"], "CIFAR100")

print("\nFine-tuning on CIFAR-10...")
model.fc = nn.Linear(model.fc.in_features, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
train(model, trainloader10, optimizer, config["epochs"], "CIFAR10")

wandb.finish()

# -----------------------------
# Now reverse: CIFAR-10 → CIFAR-100
# -----------------------------
wandb.init(project="cifar-transfer-learning", name="CIFAR10→CIFAR100")

model = torchvision.models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=config["lr"])

print("\nTraining on CIFAR-10...")
train(model, trainloader10, optimizer, config["epochs"], "CIFAR10")

print("\nFine-tuning on CIFAR-100...")
model.fc = nn.Linear(model.fc.in_features, 100).to(device)
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
train(model, trainloader100, optimizer, config["epochs"], "CIFAR100")

wandb.finish()
