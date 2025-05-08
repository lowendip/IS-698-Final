# configurable_client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import flwr as fl
import time
from pneumonia_model2 import CNN224, CNN32  # Your two models

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device, loss_fn, optimizer_class):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer_class(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        start_time = time.time()

        for epoch in range(1):
            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

        torch.save(self.model.state_dict(), "client_model.pt")
        duration = time.time() - start_time
        print(f"\n✅ Fit done in {duration:.2f}s\n")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)
                loss_sum += loss.item()
                predicted = torch.argmax(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        avg_loss = loss_sum / len(self.val_loader)
        print(f"\n📊 Eval - Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f}\n")
        return avg_loss, total, {"accuracy": accuracy}

def load_data(resolution=224, use_half=False, class_filter=None):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder("./data", transform=transform)

    if class_filter is not None:
        dataset = [d for d in dataset if d[1] == class_filter]

    if use_half:
        dataset = dataset[:len(dataset) // 2]

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    return DataLoader(train_set, batch_size=32, shuffle=True), DataLoader(val_set, batch_size=32)

def main():
    resolution = 224  # or 32
    model_type = CNN224 if resolution == 224 else CNN32
    use_half = False  # toggle half or full
    class_filter = None  # use 0 for Normal only, 1 for Pneumonia only, None for all

    train_loader, val_loader = load_data(resolution, use_half, class_filter)
    device = torch.device("cpu")
    model = model_type()

    loss_fn = nn.CrossEntropyLoss()
    optimizer_class = optim.Adam  # or optim.SGD, etc.

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FLClient(model, train_loader, val_loader, device, loss_fn, optimizer_class).to_client()
    )

if __name__ == "__main__":
    main()
