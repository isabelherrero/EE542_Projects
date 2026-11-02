
# Lab 7 - Part II: CNN with NumPy
# Step 1: Load and preprocess images

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent / "data"

def load_data(split, samples_per_class=50, seed=42):
    X, y = [], []
    rng = np.random.default_rng(seed)

    for label, category in enumerate(["cat", "dog"]):
        folder = DATA_DIR / split / category
        files = list(folder.glob("*.jpg"))
        rng.shuffle(files)

        loaded = 0

        for file in files:
            if loaded >= samples_per_class:
                break

            try:
                with Image.open(file) as img:
                    img = img.convert("RGB").resize((64, 64))
                    image = np.asarray(img, dtype=np.float32) / 255.0

                X.append(image)
                y.append(label)
                loaded += 1

            except (OSError, ValueError) as e:
                print(f"Skipping {file.name}: {e}")

        print(f"{split}/{category}: {loaded} images loaded")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Shuffle images and labels together
    indices = rng.permutation(len(X))

    return X[indices], y[indices]


# Small dataset for initial testing
X_train, y_train = load_data("train", samples_per_class=500)
X_test, y_test = load_data("test", samples_per_class=100)

print("\nDataset shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nPixel value range:")
print("Min:", X_train.min())
print("Max:", X_train.max())


# Step 2: CNN Implementation
# Based on Lab 7 PDF Page 11

# Helper functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# Initialize parameters
rng = np.random.default_rng(42)

conv_filter = rng.normal(size=(3, 3, 3)) * 0.1

# 64x64 input -> 62x62 convolution -> 31x31 pooling
W_fc = rng.normal(size=(31 * 31, 1)) * 0.01
b_fc = np.zeros((1,))


# Convolution operation
def conv2d(img, kernel):
    h, w, c = img.shape
    kh, kw, kc = kernel.shape

    out = np.zeros((h - kh + 1, w - kw + 1))

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = img[i:i + kh, j:j + kw, :]
            out[i, j] = np.sum(region * kernel)

    return out


# Max Pooling
def maxpool(img, size=2):
    h, w = img.shape
    new_h, new_w = h // size, w // size

    pooled = np.zeros((new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            pooled[i, j] = np.max(
                img[i * size:(i + 1) * size,
                    j * size:(j + 1) * size]
            )

    return pooled


# Forward Pass
def forward(img):
    global conv_filter, W_fc, b_fc

    conv_out = conv2d(img, conv_filter)
    relu_out = relu(conv_out)
    pooled = maxpool(relu_out)
    flat = pooled.flatten()

    fc_out = relu(np.dot(flat, W_fc) + b_fc)

    return fc_out, flat, pooled, relu_out


# Test Forward Pass with one image
print("\n--- CNN Forward Pass Test ---")

sample = X_train[0]

output, flat, pooled, relu_out = forward(sample)

print("Input shape:", sample.shape)
print("Convolution + ReLU shape:", relu_out.shape)
print("Max Pooling shape:", pooled.shape)
print("Flatten shape:", flat.shape)
print("Output shape:", output.shape)
print("Output value:", output)


# Step 3: Training with MSE Loss
# Based on Lab 7 PDF Page 12

from time import perf_counter

lr = 0.001
epochs = 10

print("\n--- CNN Training ---")

train_start = perf_counter()

for epoch in range(epochs):
    total_loss = 0.0
    epoch_start = perf_counter()

    for i in range(len(X_train)):
        img = X_train[i]
        label = np.array([y_train[i]])

        # Forward pass
        y_pred, flat, pooled, relu_out = forward(img)

        # Calculate loss
        loss = mse_loss(label, y_pred)
        total_loss += loss

        # Backpropagation (Fully Connected Layer only)
        grad_y = mse_grad(label, y_pred) * relu_derivative(y_pred)

        grad_W = np.outer(flat, grad_y)
        grad_b = grad_y

        # Update weights and bias
        W_fc -= lr * grad_W
        b_fc -= lr * grad_b

    avg_loss = total_loss / len(X_train)
    epoch_time = perf_counter() - epoch_start

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {avg_loss:.4f} | "
        f"Time: {epoch_time:.2f}s",
        flush=True
    )

train_time = perf_counter() - train_start

print(f"\nTotal Training Time: {train_time:.2f}s")


# Step 4: Evaluation
# Based on Lab 7 PDF Page 12

print("\n--- CNN Evaluation ---")

correct = 0
test_start = perf_counter()

for i in range(len(X_test)):
    y_pred, _, _, _ = forward(X_test[i])

    prediction = int(y_pred.item() > 0.5)

    if prediction == y_test[i]:
        correct += 1

test_time = perf_counter() - test_start
accuracy = correct / len(X_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Correct Predictions: {correct}/{len(X_test)}")
print(f"Total Test Time: {test_time:.2f}s")
print(f"Average Time per Image: {test_time / len(X_test):.4f}s")


# Step 5: Prediction Diagnostics

print("\n--- Prediction Diagnostics ---")

for name, X_data, y_data in [
    ("Train", X_train, y_train),
    ("Test", X_test, y_test)
]:
    outputs = np.array([
        forward(img)[0].item() for img in X_data
    ])

    predictions = (outputs > 0.5).astype(int)

    accuracy = np.mean(predictions == y_data)

    print(f"\n{name} Accuracy: {accuracy * 100:.2f}%")
    print("Predicted Cats:", np.sum(predictions == 0))
    print("Predicted Dogs:", np.sum(predictions == 1))
    print("Min Output:", outputs.min())
    print("Max Output:", outputs.max())
    print("Mean Output:", outputs.mean())
