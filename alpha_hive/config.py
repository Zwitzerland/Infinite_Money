import yaml
from dataclasses import dataclass

@dataclass
class Config:
    data: dict

def load_config(path: str) -> "Config":
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Config(data=d)
