
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

from datasets import load_dataset

ds = load_dataset("microsoft/cats_vs_dogs")


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


# Load images from Hugging Face instead of local folders
def load_data(dataset):
    X, y = [], []
    for item in dataset:
        img = item["image"].convert("RGB").resize((64, 64))
        X.append(np.array(img) / 255.0)
        y.append(item["labels"])
    return np.array(X), np.array(y)


# Shuffle before splitting
data = ds["train"].shuffle(seed=42)

# Start with a smaller dataset for testing
# data = data.select(range(500))

X, y = load_data(data)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training images:", len(X_train))
print("Testing images:", len(X_test))


conv_filter = np.random.randn(3, 3, 3) * 0.1
W_fc = np.random.randn(31 * 31, 1) * 0.01
b_fc = np.zeros((1,))


def conv2d(img, kernel):
    h, w, c = img.shape
    kh, kw, kc = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            region = img[i:i+kh, j:j+kw, :]
            out[i, j] = np.sum(region * kernel)
    return out


def maxpool(img, size=2):
    h, w = img.shape
    new_h, new_w = h // size, w // size
    pooled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            pooled[i, j] = np.max(img[i*size:(i+1)*size, j*size:(j+1)*size])
    return pooled


def forward(img):
    global conv_filter, W_fc, b_fc
    conv_out = conv2d(img, conv_filter)
    relu_out = relu(conv_out)
    pooled = maxpool(relu_out)
    flat = pooled.flatten()
    fc_out = relu(np.dot(flat, W_fc) + b_fc)
    return fc_out, flat, pooled, relu_out


lr = 0.001
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X_train)):
        img = X_train[i]
        label = np.array([y_train[i]])

        y_pred, flat, pooled, relu_out = forward(img)
        loss = mse_loss(label, y_pred)
        total_loss += loss

        grad_y = mse_grad(label, y_pred) * relu_derivative(y_pred)
        grad_W = np.outer(flat, grad_y)
        grad_b = grad_y

        W_fc -= lr * grad_W
        b_fc -= lr * grad_b

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train):.4f}")


correct = 0
for i in range(len(X_test)):
    y_pred, _, _, _ = forward(X_test[i])
    prediction = int(y_pred.item() > 0.5)
    if prediction == y_test[i]:
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
