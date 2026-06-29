from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMG_SIZE = 224
NUM_CLASSES = 1000

def _make_synthetic_image(rng):
    img = rng.uniform(0, 255, (3, IMG_SIZE, IMG_SIZE)).astype(np.float32)
    return img / 127.5 - 1.0

def make_synthetic(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    images = torch.stack([torch.from_numpy(_make_synthetic_image(rng)) for _ in range(n)])
    labels = torch.randint(0, NUM_CLASSES, (n,))
    return {"images": images, "labels": labels, "n_samples": n}

class EvalDataset(Dataset):
    def __init__(self, data, split="train", val_split=0.2, seed=42):
        n = data["n_samples"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split_n = int(n * (1 - val_split))
        indices = idx[:split_n] if split == "train" else idx[split_n:]
        self.images = data["images"][indices]
        self.labels = data["labels"][indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def create_dataloaders(data, batch_size=32, val_split=0.2, seed=42):
    train_ds = EvalDataset(data, "train", val_split, seed)
    val_ds = EvalDataset(data, "val", val_split, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
