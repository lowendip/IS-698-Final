import flwr as fl
from flwr.server.strategy import FedAvg

def main():
    num_clients = 5  # or set to 10 if running 10 clients

    strategy = FedAvg(
        fraction_fit=1.0,                   # Sample all clients
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"epoch_global": rnd},
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
