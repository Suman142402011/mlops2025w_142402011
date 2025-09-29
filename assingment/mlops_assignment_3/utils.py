import json
import toml

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def load_toml(path):
    return toml.load(path)
