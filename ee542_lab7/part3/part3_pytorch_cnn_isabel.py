import os
import csv
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


RANDOM_SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


class CatDogDataset(Dataset):
    def __init__(
        self,
        split_csv,
        data_dir,
        selected_set,
        transform=None
    ):
        self.samples = []
        self.transform = transform
        self.data_dir = data_dir

        with open(
            split_csv,
            "r",
            newline=""
        ) as split_file:
            reader = csv.DictReader(split_file)

            for row in reader:
                if row["set"] != selected_set:
                    continue

                label = int(row["label"])
                original_path = (
                    row["path"]
                    .replace("\\", "/")
                )

                file_name = os.path.basename(
                    original_path
                )

                class_folder = (
                    "Cat" if label == 0 else "Dog"
                )

                image_path = os.path.join(
                    data_dir,
                    class_folder,
                    file_name
                )

                self.samples.append(
                    (image_path, label)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = torch.tensor(
            [label],
            dtype=torch.float32
        )

        return image, label


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            32 * 16 * 16,
            128
        )

        self.fc2 = nn.Linear(
            128,
            1
        )

        nn.init.normal_(
                self.fc2.weight,
                mean=0.0,
                std=0.01
            )

        nn.init.constant_(
                self.fc2.bias,
                0.5
            )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x


def synchronize_gpu(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    loss_history = []

    synchronize_gpu(device)
    training_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.perf_counter()

        for images, labels in train_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs > 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        synchronize_gpu(device)

        average_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        training_accuracy = (
            correct
            / total
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        loss_history.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Training accuracy: "
            f"{training_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f} seconds"
        )

    synchronize_gpu(device)

    training_time = (
        time.perf_counter()
        - training_start
    )

    return loss_history, training_time


def evaluate_model(
    model,
    test_loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    synchronize_gpu(device)
    testing_start = time.perf_counter()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs > 0.5
            ).float()

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    synchronize_gpu(device)

    testing_time = (
        time.perf_counter()
        - testing_start
    )

    average_loss = (
        total_loss
        / len(test_loader.dataset)
    )

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        testing_time
    )


def safe_name(text):
    allowed = []

    for character in text.lower():
        if character.isalnum():
            allowed.append(character)
        else:
            allowed.append("_")

    return "".join(allowed).strip("_")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="gpu"
    )

    parser.add_argument(
        "--data-dir",
        default=(
            "data/kagglecatsanddogs_3367a/"
            "PetImages"
        )
    )

    parser.add_argument(
        "--split-file",
        default="results/dataset_split.csv"
    )

    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU:", gpu_name)
        print("CUDA version:", torch.version.cuda)
    else:
        gpu_name = "CPU"
        print(
            "WARNING: CUDA is not available. "
            "The program will run on the CPU."
        )

    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor()
    ])

    train_dataset = CatDogDataset(
        split_csv=args.split_file,
        data_dir=args.data_dir,
        selected_set="train",
        transform=transform
    )

    test_dataset = CatDogDataset(
        split_csv=args.split_file,
        data_dir=args.data_dir,
        selected_set="test",
        transform=transform
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    print("\nDataset information")
    print(
        "Training images:",
        len(train_dataset)
    )
    print(
        "Testing images:",
        len(test_dataset)
    )

    first_images, first_labels = next(
        iter(train_loader)
    )

    print(
        "First image batch shape:",
        first_images.shape
    )

    print(
        "First label batch shape:",
        first_labels.shape
    )

    model = CatDogCNN().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nModel")
    print(model)

    print("\nTraining model...")

    loss_history, training_time = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    print("\nEvaluating model...")

    (
        test_loss,
        test_accuracy,
        testing_time
    ) = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )

    print("\nPART III RESULTS")
    print("Run name:", args.run_name)
    print("Device:", device)
    print("GPU:", gpu_name)
    print(
        f"Final training loss: "
        f"{loss_history[-1]:.4f}"
    )
    print(
        f"Test loss: {test_loss:.4f}"
    )
    print(
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Total training time: "
        f"{training_time:.2f} seconds"
    )
    print(
        f"Total testing time: "
        f"{testing_time:.2f} seconds"
    )

    os.makedirs(
        "results",
        exist_ok=True
    )

    run_name = safe_name(args.run_name)

    model_path = (
        f"results/part3_{run_name}_model.pth"
    )

    results_path = (
        f"results/part3_{run_name}_results.txt"
    )

    graph_path = (
        f"results/part3_{run_name}_loss.png"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    plt.figure()

    plt.plot(
        range(1, EPOCHS + 1),
        loss_history,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        f"PyTorch CNN Training Loss: "
        f"{args.run_name}"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(graph_path)
    plt.close()

    with open(
        results_path,
        "w"
    ) as results_file:
        results_file.write(
            "Part III PyTorch CNN Results\n"
        )

        results_file.write(
            f"Run name: {args.run_name}\n"
        )

        results_file.write(
            f"Device: {device}\n"
        )

        results_file.write(
            f"GPU: {gpu_name}\n"
        )

        results_file.write(
            f"CUDA version: "
            f"{torch.version.cuda}\n"
        )

        results_file.write(
            f"Training images: "
            f"{len(train_dataset)}\n"
        )

        results_file.write(
            f"Testing images: "
            f"{len(test_dataset)}\n"
        )

        results_file.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        results_file.write(
            f"Epochs: {EPOCHS}\n"
        )

        results_file.write(
            f"Final training loss: "
            f"{loss_history[-1]:.4f}\n"
        )

        results_file.write(
            f"Test loss: "
            f"{test_loss:.4f}\n"
        )

        results_file.write(
            f"Test accuracy: "
            f"{test_accuracy * 100:.2f}%\n"
        )

        results_file.write(
            f"Total training time: "
            f"{training_time:.2f} seconds\n"
        )

        results_file.write(
            f"Total testing time: "
            f"{testing_time:.2f} seconds\n"
        )

    print("\nSaved:")
    print(model_path)
    print(results_path)
    print(graph_path)


if __name__ == "__main__":
    main()
