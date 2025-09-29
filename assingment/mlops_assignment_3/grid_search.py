import os
import torch
import torch.nn as nn
from itertools import product
from models import build_model
from train import train_one_epoch
from config import config, params, hyper, device, train_loader, num_classes

def run_grid_search(arch_name, save_model_dir="./best_models"):
    print(f"\n=== Grid search for {arch_name} ===")
    best_acc = 0.0
    best_hyper = {}
    all_results = []

    os.makedirs(save_model_dir, exist_ok=True)

    for lr, opt_name, mom in product(hyper["learning_rate"], hyper["optimizer"], hyper["momentum"]):
        print(f"\n-> Trying lr={lr}, opt={opt_name}, momentum={mom}")
        model = build_model(arch_name, num_classes=num_classes, pretrained=config["model"]["pretrained"])
        model.to(device)
        criterion = nn.CrossEntropyLoss()

        if opt_name.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=params.get(arch_name, {}).get("weight_decay", 0.0))
        elif opt_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom,
                                        weight_decay=params.get(arch_name, {}).get("weight_decay", 0.0))
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        stats = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Result -> loss={stats['loss']:.4f}, acc={stats['acc']:.4f}")

        if stats["acc"] > best_acc:
            best_acc = stats["acc"]
            best_hyper = {
                "arch": arch_name,
                "learning_rate": lr,
                "optimizer": opt_name,
                "momentum": mom,
                "accuracy": stats["acc"],
                "loss": stats["loss"]
            }

    # Save best results to TXT
    txt_path = os.path.join(save_model_dir, f"{arch_name}_best.txt")
    with open(txt_path, "w") as f:
        for k, v in best_hyper.items():
            f.write(f"{k}: {v}\n")

    print(f"\nBest hyperparameters for {arch_name}: {best_hyper}")
    print(f"Best model summary saved to: {txt_path}")
