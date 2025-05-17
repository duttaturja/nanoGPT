import torch
import os
from nanoGPT.model import GPTModel
from nanoGPT.transformer import Transformer
from nanoGPT.utils import load_data, save_model, generate_text
from data_loader import get_shakespeare
from fl_utils import get_fl_client, get_fl_server
from nanoGPT.cli_interface import get_parser

parser = get_parser()
args = parser.parse_args()

def main():
    if args.mode == "train":
        train_local()

    elif args.mode == "fl_server":
        get_fl_server(args.rounds)

    elif args.mode == "fl_client":
        get_fl_client(cid=args.cid, num_clients=args.clients, epochs=args.epochs)

    elif args.mode == "generate":
        generate()

def train_local():
    train_data, val_data, vocab_size, stoi, itos = load_data(args.data, args.block_size)

    model = GPTModel(vocab_size) if args.model == "gpt" else Transformer(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for i in range(0, len(train_data) - args.block_size, args.batch):
            xb = train_data[i:i+args.block_size]
            yb = train_data[i+1:i+1+args.block_size]
            logits, loss = model(xb.unsqueeze(0), yb.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {sum(losses)/len(losses):.4f}")
    save_model(model)

def generate():
    _, _, vocab_size, stoi, itos = load_data("input.txt", 128)
    model = GPTModel(vocab_size) if args.model == "gpt" else Transformer(vocab_size)
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()
        prompt = args.prompt
        print(generate_text(model, prompt, stoi, itos, args.length, args.temperature))
    else:
        print(f"Checkpoint {args.checkpoint} not found!")

if __name__ == "__main__":
    main()