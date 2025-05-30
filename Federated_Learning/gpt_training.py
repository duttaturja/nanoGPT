import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def train_one_epoch(model, dataloader, device, lr):
    model.train()
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        if isinstance(logits, tuple):  # <-- handle tuple output
            logits = logits[0]

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
