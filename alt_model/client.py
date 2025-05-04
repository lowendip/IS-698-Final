# client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import flwr as fl
from pneumonia_model import PneumoniaCNN

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("✅ fit() called")
        self.set_parameters(parameters)

        self.model.train()
        try:
            num_epochs = 10   
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, (images, labels) in enumerate(self.train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                avg_loss = running_loss / len(self.train_loader)
                print(f"📚 Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

            torch.save(self.model.state_dict(), "client_model.pt")
            print("💾 Model saved: client_model.pt")

        except Exception as e:
            print(f"❌ Training failed: {e}")

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_loader.dataset), {"accuracy": 0.0}  # stub

def load_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomRotation(10),      
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder("./data", transform=transform)
    print(f"🗂️ Class mapping (ImageFolder): {dataset.class_to_idx}")
    
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, _ = random_split(dataset, [train_len, val_len])
    return DataLoader(train_set, batch_size=32, shuffle=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN()
    train_loader = load_data()
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FLClient(model, train_loader, device).to_client()
    )

if __name__ == "__main__":
    main()
