# #!/usr/bin/env python3
# """
# model_pipeline.py

# Features:
# - Reads config.json (model architectures)
# - Reads params.toml (architecture-specific params)
# - Reads hyperparams.json for grid search
# - Loads pretrained ResNet variants (34,50,101,152) from torchvision
# - Runs a demo inference pass on random data
# - Performs grid search and saves best hyperparameters in .txt files
# """

# import json
# import toml
# from itertools import product
# import torch
# import torch.nn as nn
# from torchvision import models
# from torch.utils.data import Dataset, DataLoader
# import os
# from tqdm import tqdm

# # ---------------------------
# # Random Dataset
# # ---------------------------
# class RandomDataset(Dataset):
#     def __init__(self, num_samples=500, num_classes=10, img_size=224):
#         self.num_samples = num_samples
#         self.num_classes = num_classes
#         self.img_size = img_size

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         img = torch.randn(3, self.img_size, self.img_size)
#         label = torch.randint(0, self.num_classes, (1,)).item()
#         return img, label

# # ---------------------------
# # Utility functions
# # ---------------------------
# def load_json(path):
#     with open(path, "r") as f:
#         return json.load(f)

# def load_toml(path):
#     return toml.load(path)

# # ---------------------------
# # Configs
# # ---------------------------
# CONFIG_PATH = "config.json"
# PARAMS_PATH = "params.toml"
# HYPER_PATH = "hyperparams.json"

# config = load_json(CONFIG_PATH)
# params = load_toml(PARAMS_PATH)
# hyper = load_json(HYPER_PATH)["hyperparameters"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# img_size = config["data"].get("image_size", 224)
# batch_size = config["data"].get("batch_size", 32)
# num_classes = config["data"].get("num_classes", 10)
# train_subset = config["data"].get("train_subset", 500)
# val_subset = config["data"].get("val_subset", 100)

# train_loader = DataLoader(RandomDataset(train_subset, num_classes, img_size),
#                           batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(RandomDataset(val_subset, num_classes, img_size),
#                         batch_size=batch_size, shuffle=False)

# # ---------------------------
# # ResNet factory
# # ---------------------------
# RESNET_MAP = {
#     "resnet34": models.resnet34,
#     "resnet50": models.resnet50,
#     "resnet101": models.resnet101,
#     "resnet152": models.resnet152
# }

# def build_model(name, num_classes, pretrained=True):
#     if name not in RESNET_MAP:
#         raise ValueError(f"Unknown architecture: {name}")
#     weights = "IMAGENET1K_V1" if pretrained else None
#     model = RESNET_MAP[name](weights=weights)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     return model

# # ---------------------------
# # Inference demo
# # ---------------------------
# def inference_demo(arch_name):
#     print(f"\n=== Inference demo for {arch_name} ===")
#     model = build_model(arch_name, num_classes=num_classes, pretrained=config["model"]["pretrained"])
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images = images.to(device)
#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1)
#             print(f"Sample preds ({arch_name}): {preds[:8].cpu().tolist()}")
#             break

# # ---------------------------
# # Grid search + save best results as TXT
# # ---------------------------
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total = 0
#     correct = 0
#     for images, labels in tqdm(loader, desc="train"):
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         preds = outputs.argmax(dim=1)
#         total += labels.size(0)
#         correct += (preds == labels).sum().item()
#         break  # demo: only one batch
#     acc = correct / total if total > 0 else 0.0
#     return {"loss": loss.item() if 'loss' in locals() else 0.0, "acc": acc}

# def run_grid_search(arch_name, save_path="results.json", save_model_dir="./best_models"):
#     print(f"\n=== Grid search for {arch_name} ===")
#     best_acc = 0.0
#     best_hyper = {}
#     all_results = []

#     os.makedirs(save_model_dir, exist_ok=True)

#     for lr, opt_name, mom in product(hyper["learning_rate"], hyper["optimizer"], hyper["momentum"]):
#         print(f"\n-> Trying lr={lr}, opt={opt_name}, momentum={mom}")
#         model = build_model(arch_name, num_classes=num_classes, pretrained=config["model"]["pretrained"])
#         model.to(device)
#         criterion = nn.CrossEntropyLoss()

#         if opt_name.lower() == "adam":
#             optimizer = torch.optim.Adam(model.parameters(), lr=lr,
#                                          weight_decay=params.get(arch_name, {}).get("weight_decay", 0.0))
#         elif opt_name.lower() == "sgd":
#             optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom,
#                                         weight_decay=params.get(arch_name, {}).get("weight_decay", 0.0))
#         else:
#             raise ValueError(f"Unsupported optimizer: {opt_name}")

#         stats = train_one_epoch(model, train_loader, optimizer, criterion)
#         print(f"Result (demo) lr={lr}, opt={opt_name}, mom={mom} -> loss={stats['loss']:.4f}, acc={stats['acc']:.4f}")

#         all_results.append({
#             "arch": arch_name,
#             "learning_rate": lr,
#             "optimizer": opt_name,
#             "momentum": mom,
#             "loss": stats["loss"],
#             "accuracy": stats["acc"]
#         })

#         if stats["acc"] > best_acc:
#             best_acc = stats["acc"]
#             best_hyper = {
#                 "arch": arch_name,
#                 "learning_rate": lr,
#                 "optimizer": opt_name,
#                 "momentum": mom,
#                 "accuracy": stats["acc"],
#                 "loss": stats["loss"]
#             }

#     # Save best results to TXT
#     txt_path = os.path.join(save_model_dir, f"{arch_name}_best.txt")
#     with open(txt_path, "w") as f:
#         f.write(f"Best model for architecture: {arch_name}\n")
#         f.write(f"Accuracy: {best_hyper['accuracy']:.4f}\n")
#         f.write(f"Loss: {best_hyper['loss']:.4f}\n")
#         f.write(f"Learning rate: {best_hyper['learning_rate']}\n")
#         f.write(f"Optimizer: {best_hyper['optimizer']}\n")
#         f.write(f"Momentum: {best_hyper['momentum']}\n")

#     print(f"\nBest hyperparameters for {arch_name}: {best_hyper}")
#     print(f"Best model summary saved to: {txt_path}")

# # ---------------------------
# # Main entry
# # ---------------------------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Model pipeline driver")
#     parser.add_argument("--mode", choices=["inference", "grid", "all"], default="all",
#                         help="Which pipeline part to run: inference, grid, or all")
#     parser.add_argument("--arch", default=None, help="Run for a specific architecture (e.g. resnet34)")
#     args = parser.parse_args()

#     if args.mode in ("inference", "all"):
#         if args.arch:
#             inference_demo(args.arch)
#         else:
#             for arch in config["model"]["architecture"]:
#                 inference_demo(arch)

#     if args.mode in ("grid", "all"):
#         if args.arch:
#             run_grid_search(args.arch)
#         else:
#             for arch in config["model"]["architecture"]:
#                 run_grid_search(arch)

import argparse
from config import config
from inference import inference_demo
from grid_search import run_grid_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model pipeline driver")
    parser.add_argument("--mode", choices=["inference", "grid", "all"], default="all",
                        help="Which pipeline part to run: inference, grid, or all")
    parser.add_argument("--arch", default=None, help="Run for a specific architecture (e.g. resnet34)")
    args = parser.parse_args()

    if args.mode in ("inference", "all"):
        if args.arch:
            inference_demo(args.arch)
        else:
            for arch in config["model"]["architecture"]:
                inference_demo(arch)

    if args.mode in ("grid", "all"):
        if args.arch:
            run_grid_search(args.arch)
        else:
            for arch in config["model"]["architecture"]:
                run_grid_search(arch)
