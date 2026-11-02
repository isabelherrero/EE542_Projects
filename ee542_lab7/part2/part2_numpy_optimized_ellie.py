
# Lab 7 - Part II: Optimized NumPy CNN

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Configuration
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results" / "part2_optimized"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 10
LR = 0.001
SEED = 42

rng = np.random.default_rng(SEED)

# Same basic CNN architecture as the PDF
conv_filter = rng.normal(0, 0.1, (3, 3, 3)).astype(np.float32)
W_fc = rng.normal(0, 0.01, 31 * 31).astype(np.float32)
b_fc = 0.0


# Step 1: Optimized feature extraction
def extract_features(img):
    # Vectorized 3x3 convolution
    windows = np.lib.stride_tricks.sliding_window_view(
        img, (3, 3), axis=(0, 1)
    )

    conv_out = np.einsum(
        "hwcij,ijc->hw",
        windows,
        conv_filter,
        optimize=True
    )

    # ReLU
    relu_out = np.maximum(0, conv_out)

    # Vectorized 2x2 Max Pooling
    pooled = relu_out.reshape(31, 2, 31, 2).max(axis=(1, 3))

    return pooled.flatten()


# Step 2: Extract all features once
def load_features(split, seed):
    files = []

    for label, category in enumerate(["cat", "dog"]):
        folder = DATA_DIR / split / category

        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((path, label))

    shuffle_rng = np.random.default_rng(seed)
    shuffle_rng.shuffle(files)

    X = np.empty((len(files), 961), dtype=np.float32)
    y = np.empty(len(files), dtype=np.int64)

    start = perf_counter()
    valid = 0

    print(f"\nExtracting {split} features...", flush=True)

    for i, (path, label) in enumerate(files, 1):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize((64, 64))
                image = np.asarray(img, dtype=np.float32) / 255.0

            X[valid] = extract_features(image)
            y[valid] = label
            valid += 1

        except (OSError, ValueError) as e:
            print(f"Skipping {path.name}: {e}", flush=True)

        if i % 2500 == 0 or i == len(files):
            elapsed = perf_counter() - start
            print(
                f"{split}: {i}/{len(files)} checked, "
                f"{valid} loaded, {elapsed:.2f}s",
                flush=True
            )

    elapsed = perf_counter() - start

    return X[:valid], y[:valid], elapsed


program_start = perf_counter()

X_train, y_train, train_extract_time = load_features(
    "train", SEED
)
X_test, y_test, test_extract_time = load_features(
    "test", SEED + 1
)

print(f"\nTraining samples: {len(y_train)}")
print(f"Testing samples: {len(y_test)}")


# Step 3: Train fully connected layer
print("\nTraining fully connected layer...", flush=True)

losses = []
train_start = perf_counter()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for flat, label in zip(X_train, y_train):
        raw_output = float(np.dot(flat, W_fc) + b_fc)
        prediction = max(0.0, raw_output)

        error = prediction - float(label)
        total_loss += error ** 2

        # Backpropagation through output ReLU
        grad = 2.0 * error if raw_output > 0 else 0.0

        # SGD update
        W_fc -= LR * grad * flat
        b_fc -= LR * grad

    avg_loss = total_loss / len(X_train)
    losses.append(avg_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}",
        flush=True
    )

fc_train_time = perf_counter() - train_start


# Step 4: Evaluate model
def evaluate(X, y):
    outputs = np.maximum(0, X @ W_fc + b_fc)
    predictions = (outputs > 0.5).astype(int)

    accuracy = np.mean(predictions == y)

    return accuracy, predictions


eval_start = perf_counter()

train_accuracy, train_predictions = evaluate(X_train, y_train)
test_accuracy, test_predictions = evaluate(X_test, y_test)

eval_time = perf_counter() - eval_start

total_training = train_extract_time + fc_train_time
total_testing = test_extract_time + eval_time
program_time = perf_counter() - program_start


# Step 5: Print results
summary = f"""
--- PART II OPTIMIZED RESULTS ---

Training images: {len(y_train)}
Testing images: {len(y_test)}
Epochs: {EPOCHS}
Learning rate: {LR}

Initial loss: {losses[0]:.4f}
Final loss: {losses[-1]:.4f}

Train accuracy: {train_accuracy * 100:.2f}%
Test accuracy: {test_accuracy * 100:.2f}%

Training feature extraction: {train_extract_time:.2f}s
Fully connected training: {fc_train_time:.2f}s
Total training time: {total_training:.2f}s

Testing feature extraction: {test_extract_time:.2f}s
Evaluation time: {eval_time:.4f}s
Total testing time: {total_testing:.2f}s

Average test processing time: {total_testing / len(y_test):.6f}s
Complete program time: {program_time:.2f}s

Predicted cats: {np.sum(test_predictions == 0)}
Predicted dogs: {np.sum(test_predictions == 1)}
"""

print(summary)


# Step 6: Save results
with open(RESULT_DIR / "part2_results.txt", "w") as f:
    f.write(summary)

with open(RESULT_DIR / "training_loss.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Loss"])

    for epoch, loss in enumerate(losses, 1):
        writer.writerow([epoch, loss])


# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("NumPy CNN Training Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULT_DIR / "loss_curve.png", dpi=150)
plt.close()

print("Results saved to:", RESULT_DIR)


# Additional Evaluation: Confusion Matrix

cm = np.zeros((2, 2), dtype=int)

for actual, predicted in zip(y_test, test_predictions):
    cm[actual, predicted] += 1

print("\n--- Confusion Matrix ---")
print("             Pred Cat   Pred Dog")
print(f"Actual Cat   {cm[0,0]:8d}   {cm[0,1]:8d}")
print(f"Actual Dog   {cm[1,0]:8d}   {cm[1,1]:8d}")

cat_acc = cm[0,0] / cm[0].sum()
dog_acc = cm[1,1] / cm[1].sum()

print(f"Cat Accuracy: {cat_acc * 100:.2f}%")
print(f"Dog Accuracy: {dog_acc * 100:.2f}%")

np.savetxt(
    RESULT_DIR / "confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d",
    header="Predicted Cat,Predicted Dog",
    comments=""
)
