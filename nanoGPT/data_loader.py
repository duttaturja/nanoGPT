import numpy
import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("data/me.txt", "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T]
        x = (buf[:-1]).view(B, T)  # src
        y = buf[1:].view(B, T)  # tgt

        self.current_position += B * T
        if self.current_position + B * T >= len(self.tokens):
            self.current_position = 0

        return x, y
