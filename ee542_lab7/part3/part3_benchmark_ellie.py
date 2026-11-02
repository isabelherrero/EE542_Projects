
# Lab 7 Part III - CPU vs GPU Benchmark

import csv
import random
from pathlib import Path
from time import perf_counter
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results" / "part3_benchmark"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 1
BATCH_SIZE = 64
LR = 0.001
SEED = 42

TRAIN_PER_CLASS = 500
TEST_PER_CLASS = 100


# Same CNN architecture as the original Part III
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


def create_subset(dataset, per_class):
    rng = random.Random(SEED)
    indices = []

    for label in [0, 1]:
        candidates = [
            i for i, target in enumerate(dataset.targets)
            if target == label
        ]

        indices.extend(rng.sample(candidates, per_class))

    indices.sort()
    return Subset(dataset, indices)


def make_loader(dataset, shuffle, workers, pin_memory):
    options = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "generator": torch.Generator().manual_seed(SEED),
    }

    if workers > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = 2

    return DataLoader(dataset, **options)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(name, device, train_data, test_data,
              workers=0, use_amp=False):

    print(f"\n--- {name} ---", flush=True)

    torch.manual_seed(SEED)

    model = CatDogCNN().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp
    )

    train_loader = make_loader(
        train_data,
        shuffle=True,
        workers=workers,
        pin_memory=(device.type == "cuda")
    )

    test_loader = make_loader(
        test_data,
        shuffle=False,
        workers=workers,
        pin_memory=(device.type == "cuda")
    )

    # Training
    model.train()

    synchronize(device)
    train_start = perf_counter()

    losses = []

    for epoch in range(EPOCHS):
        loss_sum = torch.zeros((), device=device)

        for images, labels in train_loader:
            images = images.to(
                device, non_blocking=(device.type == "cuda")
            )
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            context = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if use_amp else nullcontext()
            )

            with context:
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.detach() * images.size(0)

        epoch_loss = (loss_sum / len(train_data)).item()
        losses.append(epoch_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Loss: {epoch_loss:.4f}",
            flush=True
        )

    synchronize(device)
    train_time = perf_counter() - train_start

    # Evaluation
    model.eval()

    correct = torch.zeros((), dtype=torch.long, device=device)

    synchronize(device)
    test_start = perf_counter()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(
                device, non_blocking=(device.type == "cuda")
            )
            labels = labels.to(device)

            context = (
                torch.amp.autocast("cuda", dtype=torch.float16)
                if use_amp else nullcontext()
            )

            with context:
                outputs = model(images)

            predictions = (outputs > 0.5).long().view(-1)
            correct += (predictions == labels).sum()

    synchronize(device)
    test_time = perf_counter() - test_start

    accuracy = correct.item() / len(test_data)

    result = {
        "Experiment": name,
        "Device": str(device),
        "AMP": use_amp,
        "DataLoader Workers": workers,
        "Training Images": len(train_data),
        "Testing Images": len(test_data),
        "Epochs": EPOCHS,
        "Batch Size": BATCH_SIZE,
        "Final Loss": round(losses[-1], 4),
        "Test Accuracy (%)": round(accuracy * 100, 2),
        "Training Time (s)": round(train_time, 2),
        "Testing Time (s)": round(test_time, 2),
        "Total Time (s)": round(train_time + test_time, 2),
    }

    print(
        f"Train: {train_time:.2f}s | "
        f"Test: {test_time:.2f}s | "
        f"Accuracy: {accuracy*100:.2f}%",
        flush=True
    )

    del model, optimizer, scaler

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    print("GPU:", torch.cuda.get_device_name(0))

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    full_train = datasets.ImageFolder(
        DATA_DIR / "train", transform=transform
    )

    full_test = datasets.ImageFolder(
        DATA_DIR / "test", transform=transform
    )

    train_data = create_subset(full_train, TRAIN_PER_CLASS)
    test_data = create_subset(full_test, TEST_PER_CLASS)

    print("Training images:", len(train_data))
    print("Testing images:", len(test_data))

    gpu = torch.device("cuda")
    cpu = torch.device("cpu")

    results = []

    # GPU control run
    results.append(
        benchmark(
            "GPU Baseline", gpu, train_data, test_data,
            workers=0, use_amp=False
        )
    )

    # Experiment 2: CPU
    results.append(
        benchmark(
            "CPU Baseline", cpu, train_data, test_data,
            workers=0, use_amp=False
        )
    )

    # Experiment 3: optimized GPU
    results.append(
        benchmark(
            "GPU Optimized", gpu, train_data, test_data,
            workers=2, use_amp=True
        )
    )

    output_file = RESULT_DIR / "benchmark_results.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n--- BENCHMARK SUMMARY ---")

    for r in results:
        print(
            f"{r['Experiment']}: "
            f"Train={r['Training Time (s)']}s, "
            f"Test={r['Testing Time (s)']}s, "
            f"Accuracy={r['Test Accuracy (%)']}%"
        )

    gpu_base = results[0]["Training Time (s)"]
    cpu_base = results[1]["Training Time (s)"]
    gpu_opt = results[2]["Training Time (s)"]

    print(f"\nCPU/GPU training speedup: {cpu_base/gpu_base:.2f}x")
    print(f"GPU optimization speedup: {gpu_base/gpu_opt:.2f}x")

    print("\nResults saved to:", output_file)


if __name__ == "__main__":
    main()
