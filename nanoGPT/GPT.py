"""
Main entry point for training both GPT and Transformer models.
"""
import argparse, torch, math
from torch.optim import AdamW
from torch.utils.data import DataLoader
from nanoGPT.utils import TextDataset, set_seed, save_checkpoint, load_checkpoint
from nanoGPT.config import GPTConfig, TransformerConfig
from nanoGPT.transformer import TransformerModel
from nanoGPT.model import GPT  # original GPT implementation in model.py

def evaluate(model, loader, device, is_transformer=False):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_transformer:
                logits = model(x, x[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            else:
                _, loss = model(x, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    ppl = math.exp(total_loss / n)
    model.train()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Train GPT or Transformer model")
    parser.add_argument("--data", default="input.txt", help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--block_size", type=int, default=128, help="Context length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--eval_interval", type=int, default=100, help="Steps between evals")
    parser.add_argument("--model", choices=["gpt", "transformer"], default="gpt", help="Model type to train")
    args = parser.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = TextDataset(args.data, args.block_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(train_ds, batch_size=args.batch_size)

    if args.model == "gpt":
        cfg = GPTConfig(vocab_size=len(train_ds.vocab), block_size=args.block_size)
        model = GPT(cfg).to(device)
        is_transformer = False
    else:
        cfg = TransformerConfig(vocab_size=len(train_ds.vocab), block_size=args.block_size)
        model = TransformerModel(cfg).to(device)
        is_transformer = True

    optim = AdamW(model.parameters(), lr=args.lr)
    step = load_checkpoint(model, optim)

    for epoch in range(args.epochs):
        for x, y in train_loader:
            step += 1
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            if is_transformer:
                logits = model(x, x[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
            else:
                _, loss = model(x, y)
            loss.backward()
            optim.step()

            if step % args.eval_interval == 0:
                ppl = evaluate(model, eval_loader, device, is_transformer)
                print(f"epoch {epoch+1}/{args.epochs} step {step} loss {loss.item():.4f} PPL {ppl:.1f}")

        save_checkpoint(model, optim, step)

if __name__ == "__main__":
    main()