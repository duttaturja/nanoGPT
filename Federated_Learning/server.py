import flwr as fl
from Federated_Learning.config import ROUNDS

def start_server():
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=4,
        fraction_fit=1.0
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
