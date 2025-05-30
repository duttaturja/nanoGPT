import torch
import tiktoken
import torch.nn.functional as F
from nanoGPT.transformer import GPT
from nanoGPT.config import GPTConfig
from transformers import GPT2Tokenizer
import os
import time

num_return_sequences = 1
max_length = 300

device = "cpu"

model = GPT.from_pretrained("gpt2")
model.eval()
model.to(device)
model.load_state_dict(torch.load("./model.pth"))
print("loading model...")
print("loaded model successfully!")
time.sleep(2)
os.system("cls" if os.name == "nt" else "clear")
prompt = input("Me: ")

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(prompt)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5,8)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the next token(logits)
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[0][:, -1, :]  # (B, vocab_size)
        # sample from the logits
        probs = F.softmax(logits, dim=-1)
        # get the top-k probabilities and their indices
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        ix = torch.multinomial(topk_probs, 1)  # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)  # (B, T+1)

# printing the generated texts
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">nanoGPT: ", decoded)
