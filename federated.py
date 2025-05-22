import flwr as fl
from torch.utils.data import DataLoader

from Federated_Learning.data_utils import load_and_partition_dataset
from Federated_Learning.config import NUM_CLIENTS, BATCH_SIZE
from Federated_Learning.client import GPTClient


# --------------------------------------------------------------------------- #
# Flower expects client_fn(cid: str) — we'll keep that simple, no Context.    #
# --------------------------------------------------------------------------- #
partitions, VOCAB_SIZE = load_and_partition_dataset(
    "Dataset/html.txt", NUM_CLIENTS, block_size=128
)


def client_fn(cid: str):
    cid_int = int(cid)  # Flower still passes a string
    trainloader = DataLoader(
        partitions[cid_int], batch_size=BATCH_SIZE, shuffle=True
    )
    return GPTClient(trainloader, VOCAB_SIZE)


if __name__ == "__main__":
    fl.simulation.start_simulation(          # still works, ignore deprecation msg
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        client_resources={"num_cpus": 1, "num_gpus": 0},  # tweak if you have GPU
    )
