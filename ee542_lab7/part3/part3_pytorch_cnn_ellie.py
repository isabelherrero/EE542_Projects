
# Lab 7 - Part III: GPU Acceleration with PyTorch

import csv
from pathlib import Path
from time import perf_counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Configuration
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results" / "part3_gpu"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

if device.type != "cuda":
    raise RuntimeError("CUDA GPU not detected. Check PyTorch installation.")

print("GPU:", torch.cuda.get_device_name(0))


# Step 1: Load dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=transform
)

print("Classes:", train_dataset.class_to_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Training images:", len(train_dataset))
print("Testing images:", len(test_dataset))


# Step 2: Define CNN model
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


model = CatDogCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Step 3: Training
print("\n--- GPU Training ---")

losses = []
torch.cuda.synchronize()
train_start = perf_counter()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    epoch_loss = total_loss / len(train_dataset)
    losses.append(epoch_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {epoch_loss:.4f}",
        flush=True
    )

torch.cuda.synchronize()
train_time = perf_counter() - train_start


# Step 4: Evaluation
def evaluate(loader):
    model.eval()

    correct = 0
    total = 0
    cm = torch.zeros((2, 2), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = (outputs > 0.5).long().view(-1).cpu()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            indices = labels.long() * 2 + preds
            cm += torch.bincount(
                indices, minlength=4
            ).reshape(2, 2)

    return correct / total, cm


print("\n--- GPU Evaluation ---")

train_accuracy, _ = evaluate(train_loader)

torch.cuda.synchronize()
test_start = perf_counter()

test_accuracy, cm = evaluate(test_loader)

torch.cuda.synchronize()
test_time = perf_counter() - test_start

cat_accuracy = cm[0, 0].item() / cm[0].sum().item()
dog_accuracy = cm[1, 1].item() / cm[1].sum().item()


# Step 5: Results
summary = f"""
--- PART III GPU RESULTS ---

GPU: {torch.cuda.get_device_name(0)}
Training images: {len(train_dataset)}
Testing images: {len(test_dataset)}
Epochs: {EPOCHS}
Batch size: {BATCH_SIZE}
Learning rate: {LR}

Initial loss: {losses[0]:.4f}
Final loss: {losses[-1]:.4f}

Training accuracy: {train_accuracy * 100:.2f}%
Test accuracy: {test_accuracy * 100:.2f}%

Training time: {train_time:.2f}s
Testing time: {test_time:.2f}s
Average test time/image: {test_time / len(test_dataset):.6f}s

Cat accuracy: {cat_accuracy * 100:.2f}%
Dog accuracy: {dog_accuracy * 100:.2f}%

Confusion Matrix:
{cm.numpy()}
"""

print(summary)


# Step 6: Save results
with open(RESULT_DIR / "part3_results.txt", "w") as f:
    f.write(summary)

with open(RESULT_DIR / "training_loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss"])

    for epoch, loss in enumerate(losses, 1):
        writer.writerow([epoch, loss])

with open(RESULT_DIR / "confusion_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Actual", "Predicted Cat", "Predicted Dog"])
    writer.writerow(["Cat", cm[0, 0].item(), cm[0, 1].item()])
    writer.writerow(["Dog", cm[1, 0].item(), cm[1, 1].item()])

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("PyTorch CNN Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULT_DIR / "loss_curve.png", dpi=150)
plt.close()

torch.save(model.state_dict(), RESULT_DIR / "cnn_model.pth")

print("Results saved to:", RESULT_DIR)
