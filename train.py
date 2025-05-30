import torch
from torch.utils.data import DataLoader
from nanoGPT.config import GPTConfig
from nanoGPT.transformer import GPT
from nanoGPT.data_loader import DataLoaderLite
import torch.optim as optim
from nanoGPT.dataset import TokenDataset

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
with open("./Dataset/me.txt", "r", encoding="utf-8") as f:
    texts = f.read()

texts = [texts]

token = TokenDataset(tokenizer, texts, block_size=128)
loader = DataLoader(token, batch_size=16, shuffle=True)
# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPTConfig(vocab_size=50257)
model = GPT(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for epoch in range(50):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        input_seq = x[:, :-1]
        target_seq = x[:, 1:]
        logits, loss = model(
            input_seq, input_seq
        )  # feeding input twice to match (src, tgt)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), target_seq.contiguous().view(-1)
        )

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
