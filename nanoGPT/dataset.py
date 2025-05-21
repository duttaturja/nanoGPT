import torch
from torch.utils.data import Dataset

class TokenDataset(Dataset):
    def __init__(self, tokenizer, texts, block_size):
        self.data = []
        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text)
            if tokens is None:
                print(f"Warning: text {i} is empty after tokenization")
            else:
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = torch.tensor(self.data[index])
        return item[:-1], item[1:]  # src, tgt