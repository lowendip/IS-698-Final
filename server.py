import torch
from models import CNN_32, CNN_224
from train_test import train, test
import matplotlib.pyplot as plt
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Metrics, Context
from load_datasets import load_datasets
from flwr.client import Client, ClientApp, NumPyClient
from flwr.simulation import run_simulation
from client import client_fn
from flwr.server.strategy import FedAvg, FedOpt, FedProx

model_save_name = "client_model.pt"
NUM_CLIENTS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on {DEVICE}")
def evaluate_fn(server_round, parameters, config):
    if server_round>0:
        #Loads current global model weights
        net = CNN_224().to(DEVICE)
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        net.eval()
        #Preforms test on full dataset and all client datasets
        print("Test set (loss, accuracy): " + str(test(net,load_datasets(partition_id=0)[2], DEVICE)))
        torch.save(net.state_dict(), model_save_name)


# Create FedAvg strategy
strategy = FedProx(
    proximal_mu=0.5,
#strategy = FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=int(NUM_CLIENTS/2),  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_fn=evaluate_fn
)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use the settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)



trainloader, _, _ = load_datasets(partition_id=0)
batch = next(iter(trainloader))
images, labels = batch["image"], batch['label']
print((images.size()))
# Reshape and convert images to a NumPy array
# matplotlib requires images with the shape (height, width, 3)
images = images.permute(0, 2, 3, 1).numpy()
# Denormalize
images = images / 2 + 0.5

# Create a figure and a grid of subplots
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

# Loop over the images and plot them
for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    #ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
    ax.axis("off")

# Show the plot

# Create the ServerApp
server = ServerApp(server_fn=server_fn)


# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
#backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

# When running on GPU, assign an entire GPU for each client
#if DEVICE == "cuda":
#    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_CLIENTS,
#    backend_config=backend_config
)
fig.tight_layout()
plt.show()

#This application uses the getting started tutorial from the Flower website as a baseline for the code: https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
