import torch
from torchvision import transforms
from PIL import Image
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt
from pneumonia_model import PneumoniaCNN

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("client_model.pt", map_location=device))
model.eval()

# Ask user to select an image
Tk().withdraw()  
image_path = filedialog.askopenfilename(title="Select Chest X-ray Image")

if not image_path:
    print("❌ No image selected.")
    exit()

# Preprocess image
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, 1).item()

labels = ["Normal", "Pneumonia"]
prediction = labels[predicted_class]

# Show result
print(f"📸 Image: {image_path}")
print(f"🧠 Prediction: {prediction}")

plt.imshow(image)
plt.title(f"Prediction: {prediction}")
plt.axis("off")
plt.show()
