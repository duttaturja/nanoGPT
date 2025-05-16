import torch, random, numpy as np, os
from torch.utils.data import Dataset

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class TextDataset(Dataset):
    def __init__(self, path, block_size):
        text = open(path, "r", encoding="utf-8").read()
        self.vocab = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.inputs = data[:-1]
        self.targets = data[1:]
        self.block_size = block_size

    def __len__(self):
        return len(self.inputs) - self.block_size

    def __getitem__(self, idx):
        ix = self.inputs[idx : idx + self.block_size]
        tx = self.targets[idx : idx + self.block_size]
        return ix, tx

def save_checkpoint(model, optimizer, step, path="ckpt.pt"):
    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(), "step": step}, path)

def load_checkpoint(model, optimizer, path="ckpt.pt"):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        return ckpt["step"]
    return 0