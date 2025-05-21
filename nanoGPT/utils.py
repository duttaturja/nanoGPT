import torch
import torch.nn as nn
import math

class TanhGELU(nn.Module): 
    def forward(self,input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0/ math.pi) * (input + 0.44715 * torch.pow(input, 3.0))))
    
#--------------------------------------------Training and loss calculation--------------------------------------------

from .transformer import GPT
from .config import GPTConfig
from .data_loader import DataLoaderLite

num_return_sequences = 5
max_length = 30

# auto device detection
import time

device = "cpu"
if torch.cuda.is_available():
    device = "cuda" # gpu device
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps" # Mac device
print(f"running on device: {device}")

# load the pretrained model from huggingface
model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)
print("didnt fucking crash! LFG")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# gradient accumulation
total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 4 # micro batch size
T = 1024 # sequence size
assert total_batch_size % (B * T) == 0, "Make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# getting the dataloaderlite in the action
train_loader = DataLoaderLite(B=B, T=T)

# tf32 precision for 300ms training time
torch.set_float32_matmul_precision('high')

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
# logits, loss = model(x, y)

# learning rate scheduler
max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) linear decay from warmup_steps to max_steps
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay from max_lr to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
    # import code; code.interact(local=locals())
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learing rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if (device == 'cuda'):
        torch.cuda.synchronize() # wait for the GPU to finish work and sync with CPU
    t1 = time.time()
    dt = (t1-t0)  # time difference in seconds
    training_processed = (train_loader.B * train_loader.T * grad_accum_steps) # total tokens processed in this batch (B * T)
    tokens_per_sec = training_processed / dt
    print(f"step {step:2d} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
