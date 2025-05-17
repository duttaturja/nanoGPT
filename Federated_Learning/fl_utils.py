import torch, numpy as np

def get_parameters(model):
    """Return model parameters as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, params):
    """Load NumPy weights into the model."""
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), params):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)