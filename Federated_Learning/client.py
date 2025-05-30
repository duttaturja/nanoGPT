import flwr as fl
import torch
from Federated_Learning.gpt_training import train_one_epoch
from Federated_Learning.config import LR, DEVICE
from nanoGPT.config import GPTConfig
from nanoGPT.transformer import GPT


class GPTClient(fl.client.NumPyClient):
    def __init__(self, trainloader, vocab_size):
        self.trainloader = trainloader
        self.model = GPT(
            GPTConfig(
                vocab_size=vocab_size, block_size=128, n_layer=4, n_head=4, n_embd=128
            )
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_one_epoch(self.model, self.trainloader, DEVICE, LR)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader.dataset), {}  # dummy eval
