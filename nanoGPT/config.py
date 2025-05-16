from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int = 256
    d_model: int = 512
    n_head: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1