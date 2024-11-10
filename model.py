import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Step 1: Load and Inspect the Dataset
ds = load_dataset("Mahadih534/Chest_CT-Scan_images-Dataset", split="train")

# Filter out any labels that are outside the expected range
num_classes = 4  # Assuming the dataset should only have 4 classes
filtered_ds = [example for example in ds if example["label"] < num_classes]
print(f"Filtered dataset length: {len(filtered_ds)}")

# Step 2: Define Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB if needed
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 3: Custom Dataset Class
class ChestCTScanDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label

# Step 4: Create DataLoader
chest_ct_scan_dataset = ChestCTScanDataset(filtered_ds, transform=transform)
train_loader = DataLoader(chest_ct_scan_dataset, batch_size=32, shuffle=True)

# Step 5: Define Model Architecture (ResNet18 with Custom Classification Head)
class CTScanClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CTScanClassifier, self).__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Instantiate the model, move to GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTScanClassifier(num_classes=num_classes)
model = model.to(device)

# Step 6: Define Loss Function, Optimizer, and Metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Metric Calculation Function
def calculate_metrics(preds, labels):
    preds = torch.argmax(preds, dim=1)
    accuracy = accuracy_score(labels.cpu(), preds.cpu())
    precision = precision_score(labels.cpu(), preds.cpu(), average='weighted')
    recall = recall_score(labels.cpu(), preds.cpu(), average='weighted')
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
    return accuracy, precision, recall, f1

# Step 7: Training Loop
def train_model(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        all_preds = []
        all_labels = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.append(outputs)
            all_labels.append(labels)

        avg_loss = total_loss / len(train_loader)
        accuracy, precision, recall, f1 = calculate_metrics(torch.cat(all_preds), torch.cat(all_labels))

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

# Step 8: Save Model Function
def save_model(model, filename="ct_scan_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

# Step 9: Train and Save the Model
train_model(model, train_loader, num_epochs=50)
save_model(model)
