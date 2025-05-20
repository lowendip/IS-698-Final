import torch
from models import CNN_32, CNN_224
from train_test import train, test
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Metrics, Context
from load_datasets import load_datasets
from flwr.client import ClientApp
from flwr.simulation import run_simulation
from client import client_fn
from flwr.server.strategy import FedAvg, FedOpt, FedProx

model_save_name = "client_model.pt"
NUM_CLIENTS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
def evaluate_fn(server_round, parameters, config):
    if server_round>0:
        # Loads current global model weights
        net = CNN_32().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        # Preforms test on full dataset and all client datasets
        print("Test set (loss, accuracy): " + str(test(net,load_datasets(partition_id=-1)[2], DEVICE)))
        #S aves model as model_save_name
        torch.save(net.state_dict(), model_save_name)

def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    #Calculate average loss and accuracy for local
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"loss":sum(losses)/sum(examples), "accuracy": sum(accuracies)/sum(examples)}


# Create FedAvg strategy
#strategy = FedProx(
#    proximal_mu=0.5,
strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=int(NUM_CLIENTS),  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_fn=evaluate_fn,
    evaluate_metrics_aggregation_fn=weighted_average
)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=10)

    return ServerAppComponents(strategy=strategy, config=config)

# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
)

#This application uses the getting started tutorial from the Flower website as a baseline for the code: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
#The weighted average was made using the tutorial from this link: https://medium.com/@entrepreneurbilal10/federated-learning-95d7a6435f08

