import torch
from torch.utils.data import DataLoader
from datasets import RandomDataset
from utils import load_json, load_toml

CONFIG_PATH = "config.json"
PARAMS_PATH = "params.toml"
HYPER_PATH = "hyperparams.json"

config = load_json(CONFIG_PATH)
params = load_toml(PARAMS_PATH)
hyper = load_json(HYPER_PATH)["hyperparameters"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img_size = config["data"].get("image_size", 224)
batch_size = config["data"].get("batch_size", 32)
num_classes = config["data"].get("num_classes", 10)
train_subset = config["data"].get("train_subset", 500)
val_subset = config["data"].get("val_subset", 100)

train_loader = DataLoader(RandomDataset(train_subset, num_classes, img_size),
                          batch_size=batch_size, shuffle=True)
val_loader = DataLoader(RandomDataset(val_subset, num_classes, img_size),
                        batch_size=batch_size, shuffle=False)
