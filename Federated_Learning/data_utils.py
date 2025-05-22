import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple


class SimpleDataset(Dataset):
    def __init__(self, data, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ---------- Helpers ----------------------------------------------------------


def _balanced_lengths(total: int, k: int) -> List[int]:
    """Return k lengths that add up to total, differing by at most 1."""
    base, rem = divmod(total, k)
    return [base + 1 if i < rem else base for i in range(k)]


def load_and_partition_dataset(
    path: str, num_clients: int, block_size: int
) -> Tuple[list, int]:
    """Return (partitions, vocab_size)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    encoded = [stoi[c] for c in text]

    dataset = SimpleDataset(encoded, block_size)
    lengths = _balanced_lengths(len(dataset), num_clients)
    partitions = random_split(dataset, lengths)

    return partitions, len(chars)
