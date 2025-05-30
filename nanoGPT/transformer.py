import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import math
import torch.optim as optim

from .model import Block
from .config import GPTConfig


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size {self.config.block_size}"
        )
        # forward the token and position embeddings
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # shape (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], (
            f"Model type {model_type} not supported"
        )
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": dict(
                n_layer=12, n_head=12, n_embd=768, block_size=768
            ),  # 124M params
            "gpt2-medium": dict(
                n_layer=24, n_head=16, n_embd=1024, block_size=1024
            ),  # 355M params
            "gpt2-large": dict(
                n_layer=36, n_head=20, n_embd=1280, block_size=1280
            ),  # 774M params
            "gpt2-xl": dict(
                n_layer=48, n_head=25, n_embd=1600, block_size=1600
            ),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model
        sd_keys = sd_keys = sd.state_dict().keys()
        sd_keys = [k for k in sd.state_dict().keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys) == len(sd_keys_hf), (
            f"Mismatch in number of keys: {len(sd_keys)} vs {len(sd_keys_hf)}"
        )
        for k in sd_keys_hf:
            if any(k.endswith(t) for t in transposed):
                assert sd_hf[k].shape[::-1] == sd.state_dict()[k].shape
                with torch.no_grad():
                    sd.state_dict()[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd.state_dict()[k].shape
                with torch.no_grad():
                    sd.state_dict()[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_group = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device

        optimizer = optim.AdamW(optim_group, lr=learning_rate)
        return optimizer
