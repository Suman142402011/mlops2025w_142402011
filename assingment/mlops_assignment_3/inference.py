import torch
from models import build_model
from config import device, val_loader, num_classes, config

def inference_demo(arch_name):
    print(f"\n=== Inference demo for {arch_name} ===")
    model = build_model(arch_name, num_classes=num_classes,
                        pretrained=config["model"]["pretrained"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            print(f"Sample preds ({arch_name}): {preds[:8].cpu().tolist()}")
            break
