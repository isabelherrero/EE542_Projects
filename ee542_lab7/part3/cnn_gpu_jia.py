
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

torch.manual_seed(42)
np.random.seed(42)

ds = load_dataset("microsoft/cats_vs_dogs")

data = ds["train"].shuffle(seed=42)
#data = data.select(range(1000))

split = int(0.8 * len(data))
train_data = data.select(range(split))
test_data = data.select(range(split, len(data)))


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])



class CatDogDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img = item["image"].convert("RGB")
        img = self.transform(img)

        label = torch.tensor([item["labels"]], dtype=torch.float32)

        return img, label


train_set = CatDogDataset(train_data, transform)
test_set = CatDogDataset(test_data, transform)

train_loader = DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_set, batch_size=32, shuffle=False, num_workers=0
)

print("Training images:", len(train_set))
print("Testing images:", len(test_set))

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)
print("Label shape:", labels.shape)



class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x


model = CatDogCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

lr = 0.001
epochs = 10

def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def train():
    model.train()
    losses = []

    sync()
    start = time.perf_counter()

    for epoch in range(epochs):
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            y_pred = model(images)
            loss = criterion(y_pred, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_set)
        losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

    sync()
    train_time = time.perf_counter() - start

    return losses, train_time


def evaluate(loader):
    model.eval()

    correct = 0
    total = 0

    sync()
    start = time.perf_counter()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            y_pred = model(images)
            predictions = (y_pred > 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    sync()
    elapsed = time.perf_counter() - start

    accuracy = correct / total
    ms_per_image = elapsed / total * 1000
    throughput = total / elapsed

    return accuracy, elapsed, ms_per_image, throughput


def main():
    os.makedirs("results_relu", exist_ok=True)

    print("\nStarting GPU training...")

    losses, train_time = train()

    print(f"\nTotal training time: {train_time:.4f} seconds")
    print(f"Average time per epoch: {train_time/epochs:.4f} seconds")

    print("\nEvaluating test images...")

    # Warm-up evaluation
    evaluate(test_loader)

    accuracy, test_time, ms, speed = evaluate(test_loader)

    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Inference time: {test_time:.4f} seconds")
    print(f"Time per image: {ms:.4f} ms")
    print(f"Throughput: {speed:.2f} images/sec")


    test_sizes = [50, 100, 200, len(test_set)]
    runs = 3
    results = []

    for n in test_sizes:
        if n > len(test_set):
            continue

        subset = torch.utils.data.Subset(test_set, range(n))
        loader = DataLoader(
            subset, batch_size=32, shuffle=False, num_workers=0
        )

        times = []
        acc = 0

        for r in range(runs):
            acc, elapsed, _, _ = evaluate(loader)
            times.append(elapsed)

        avg_time = np.mean(times)
        avg_ms = avg_time / n * 1000
        avg_speed = n / avg_time

        print(f"\nTest images: {n}")
        print(f"Accuracy: {acc*100:.2f}%")
        print(f"Average inference time: {avg_time:.4f} seconds")
        print(f"Time per image: {avg_ms:.4f} ms")
        print(f"Throughput: {avg_speed:.2f} images/sec")

        results.append([n, acc*100, avg_time, avg_ms, avg_speed])

    # ========== SAVE RESULTS ==========

    with open("results_relu/gpu_training_loss.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])

        for i, loss in enumerate(losses):
            writer.writerow([i + 1, loss])

    with open("results_relu/gpu_evaluation_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_images",
            "accuracy_percent",
            "inference_time_sec",
            "ms_per_image",
            "images_per_second"
        ])
        writer.writerows(results)

    with open("results_relu/gpu_summary.txt", "w") as f:
        f.write("PyTorch CNN Results\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Training images: {len(train_set)}\n")
        f.write(f"Testing images: {len(test_set)}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Training time: {train_time:.4f} s\n")
        f.write(f"Final loss: {losses[-1]:.6f}\n")
        f.write(f"Test accuracy: {accuracy*100:.2f}%\n")

        for n, acc, t, ms, speed in results:
            f.write(
                f"{n} images: accuracy={acc:.2f}%, "
                f"time={t:.4f}s, throughput={speed:.2f} img/s\n"
            )

    torch.save(model.state_dict(), "results_relu/cnn_gpu.pth")

    print("\nResults saved in results_relu/")


if __name__ == "__main__":
    main()
