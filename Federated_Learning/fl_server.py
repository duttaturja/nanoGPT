import flwr as fl
from flwr.server.strategy import FedAvg

strategy = FedAvg(min_fit_clients=3, min_eval_clients=3, min_available_clients=3)
fl.server.start_server(config={"num_rounds": 5}, strategy=strategy)