import os
import csv
import time
import numpy as np
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt


DATA_DIR = r"data\kagglecatsanddogs_3367a\PetImages"
TEST_IMAGES_PER_CLASS = 2500
IMAGE_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def find_valid_images(folder_path):
    valid_paths = []

    files = [
        file_name
        for file_name in os.listdir(folder_path)
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    for index, file_name in enumerate(files):
        image_path = os.path.join(
            folder_path,
            file_name
        )

        try:
            with Image.open(image_path) as image:
                image.verify()

            valid_paths.append(image_path)

        except (
            UnidentifiedImageError,
            OSError,
            ValueError
        ):
            print(
                "Skipping damaged image:",
                image_path
            )

        if (index + 1) % 2500 == 0:
            print(
                f"Checked {index + 1}/{len(files)} "
                f"files in {os.path.basename(folder_path)}"
            )

    return valid_paths


def create_dataset_split(data_dir):
    rng = np.random.default_rng(RANDOM_SEED)

    cat_folder = os.path.join(data_dir, "Cat")
    dog_folder = os.path.join(data_dir, "Dog")

    print("Checking cat images...")
    cat_paths = find_valid_images(cat_folder)

    print("\nChecking dog images...")
    dog_paths = find_valid_images(dog_folder)

    rng.shuffle(cat_paths)
    rng.shuffle(dog_paths)

    cat_test = cat_paths[:TEST_IMAGES_PER_CLASS]
    dog_test = dog_paths[:TEST_IMAGES_PER_CLASS]

    cat_train = cat_paths[TEST_IMAGES_PER_CLASS:]
    dog_train = dog_paths[TEST_IMAGES_PER_CLASS:]

    train_data = (
        [(path, 0) for path in cat_train]
        + [(path, 1) for path in dog_train]
    )

    test_data = (
        [(path, 0) for path in cat_test]
        + [(path, 1) for path in dog_test]
    )

    rng.shuffle(train_data)
    rng.shuffle(test_data)

    return train_data, test_data


def save_split(train_data, test_data):
    os.makedirs("results", exist_ok=True)

    with open(
        "results/dataset_split.csv",
        "w",
        newline=""
    ) as split_file:
        writer = csv.writer(split_file)

        writer.writerow([
            "set",
            "path",
            "label"
        ])

        for path, label in train_data:
            writer.writerow([
                "train",
                os.path.relpath(path),
                label
            ])

        for path, label in test_data:
            writer.writerow([
                "test",
                os.path.relpath(path),
                label
            ])


def load_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image_array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    return image_array


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float32)


def mse_loss(y_true, y_pred):
    return np.mean(
        (y_true - y_pred) ** 2
    )


def mse_grad(y_true, y_pred):
    return (
        2 * (y_pred - y_true)
        / y_true.size
    )


def conv2d(image, kernel):
    windows = np.lib.stride_tricks.sliding_window_view(
        image,
        (3, 3),
        axis=(0, 1)
    )

    kernel_ordered = kernel.transpose(2, 0, 1)

    output = np.sum(
        windows * kernel_ordered,
        axis=(2, 3, 4)
    )

    return output.astype(np.float32)


def maxpool(image, size=2):
    height = (
        image.shape[0]
        - image.shape[0] % size
    )

    width = (
        image.shape[1]
        - image.shape[1] % size
    )

    cropped = image[:height, :width]

    pooled = cropped.reshape(
        height // size,
        size,
        width // size,
        size
    ).max(axis=(1, 3))

    return pooled


def extract_features(
    dataset,
    conv_filter,
    dataset_name
):
    feature_size = 31 * 31

    features = np.zeros(
        (len(dataset), feature_size),
        dtype=np.float32
    )

    labels = np.zeros(
        len(dataset),
        dtype=np.float32
    )

    start_time = time.perf_counter()

    for index, (image_path, label) in enumerate(dataset):
        image = load_image(image_path)

        conv_output = conv2d(
            image,
            conv_filter
        )

        relu_output = relu(conv_output)
        pooled_output = maxpool(relu_output)

        features[index] = pooled_output.flatten()
        labels[index] = label

        if (
            (index + 1) % 500 == 0
            or index + 1 == len(dataset)
        ):
            elapsed = time.perf_counter() - start_time

            print(
                f"{dataset_name}: "
                f"{index + 1}/{len(dataset)} images "
                f"processed in {elapsed:.2f} seconds"
            )

    elapsed_time = time.perf_counter() - start_time

    return features, labels, elapsed_time


def forward(flat, W_fc, b_fc):
    linear_output = (
        np.dot(flat, W_fc)
        + b_fc
    )

    prediction = relu(linear_output)

    return prediction


overall_start = time.perf_counter()

print("Creating the train/test split...")

split_start = time.perf_counter()

train_data, test_data = create_dataset_split(
    DATA_DIR
)

split_time = time.perf_counter() - split_start

save_split(train_data, test_data)

print("\nDATASET SPLIT")
print("Training images:", len(train_data))
print("Testing images:", len(test_data))
print(
    f"Split creation time: {split_time:.2f} seconds"
)
print(
    "Split saved to results/dataset_split.csv"
)


np.random.seed(RANDOM_SEED)

conv_filter = (
    np.random.randn(3, 3, 3).astype(np.float32)
    * 0.1
)


print("\nExtracting training features...")

train_features, y_train, train_feature_time = (
    extract_features(
        train_data,
        conv_filter,
        "Training"
    )
)


print("\nExtracting testing features...")

test_features, y_test, test_feature_time = (
    extract_features(
        test_data,
        conv_filter,
        "Testing"
    )
)


np.savez_compressed(
    "results/numpy_features.npz",
    train_features=train_features,
    y_train=y_train,
    test_features=test_features,
    y_test=y_test,
    conv_filter=conv_filter
)


print("\nFeature arrays saved.")

W_fc = (
    np.random.randn(
        train_features.shape[1],
        1
    ).astype(np.float32)
    * 0.01
)

b_fc = np.zeros(
    (1,),
    dtype=np.float32
)

loss_history = []

print("\nTraining fully connected layer...")

training_start = time.perf_counter()

for epoch in range(EPOCHS):
    total_loss = 0.0

    indexes = np.random.permutation(
        len(train_features)
    )

    for index in indexes:
        flat = train_features[index]

        label = np.array(
            [y_train[index]],
            dtype=np.float32
        )

        prediction = forward(
            flat,
            W_fc,
            b_fc
        )

        loss = mse_loss(
            label,
            prediction
        )

        total_loss += loss

        gradient_y = (
            mse_grad(label, prediction)
            * relu_derivative(prediction)
        )

        gradient_W = np.outer(
            flat,
            gradient_y
        )

        gradient_b = gradient_y

        W_fc -= (
            LEARNING_RATE
            * gradient_W
        )

        b_fc -= (
            LEARNING_RATE
            * gradient_b
        )

    average_loss = (
        total_loss
        / len(train_features)
    )

    loss_history.append(average_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {average_loss:.4f}"
    )

fc_training_time = (
    time.perf_counter()
    - training_start
)

total_training_time = (
    train_feature_time
    + fc_training_time
)


print("\nEvaluating unseen test images...")

evaluation_start = time.perf_counter()

correct = 0

for index in range(len(test_features)):
    prediction = forward(
        test_features[index],
        W_fc,
        b_fc
    )

    predicted_class = int(
        prediction.item() > 0.5
    )

    if predicted_class == int(y_test[index]):
        correct += 1

evaluation_time = (
    time.perf_counter()
    - evaluation_start
)

accuracy = (
    correct
    / len(test_features)
)

total_testing_time = (
    test_feature_time
    + evaluation_time
)

overall_time = (
    time.perf_counter()
    - overall_start
)


print("\nPART II RESULTS")
print(
    f"Training images: {len(train_features)}"
)
print(
    f"Testing images: {len(test_features)}"
)
print(
    f"Final loss: {loss_history[-1]:.4f}"
)
print(
    f"Test accuracy: {accuracy * 100:.2f}%"
)
print(
    "Training feature extraction time: "
    f"{train_feature_time:.2f} seconds"
)
print(
    "Fully connected training time: "
    f"{fc_training_time:.2f} seconds"
)
print(
    "Total training time: "
    f"{total_training_time:.2f} seconds"
)
print(
    "Testing feature extraction time: "
    f"{test_feature_time:.2f} seconds"
)
print(
    "Evaluation time: "
    f"{evaluation_time:.4f} seconds"
)
print(
    "Total testing time: "
    f"{total_testing_time:.2f} seconds"
)
print(
    "Complete program time: "
    f"{overall_time:.2f} seconds"
)


plt.figure()

plt.plot(
    range(1, EPOCHS + 1),
    loss_history,
    marker="o"
)

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Part II NumPy CNN Training Loss")
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/part2_numpy_loss.png"
)

plt.close()


with open(
    "results/part2_numpy_results.txt",
    "w"
) as results_file:
    results_file.write(
        "Part II NumPy CNN Results\n"
    )

    results_file.write(
        f"Training images: "
        f"{len(train_features)}\n"
    )

    results_file.write(
        f"Testing images: "
        f"{len(test_features)}\n"
    )

    results_file.write(
        f"Epochs: {EPOCHS}\n"
    )

    results_file.write(
        f"Final loss: "
        f"{loss_history[-1]:.4f}\n"
    )

    results_file.write(
        f"Test accuracy: "
        f"{accuracy * 100:.2f}%\n"
    )

    results_file.write(
        "Training feature extraction time: "
        f"{train_feature_time:.2f} seconds\n"
    )

    results_file.write(
        "Fully connected training time: "
        f"{fc_training_time:.2f} seconds\n"
    )

    results_file.write(
        "Total training time: "
        f"{total_training_time:.2f} seconds\n"
    )

    results_file.write(
        "Testing feature extraction time: "
        f"{test_feature_time:.2f} seconds\n"
    )

    results_file.write(
        "Evaluation time: "
        f"{evaluation_time:.4f} seconds\n"
    )

    results_file.write(
        "Total testing time: "
        f"{total_testing_time:.2f} seconds\n"
    )

    results_file.write(
        "Complete program time: "
        f"{overall_time:.2f} seconds\n"
    )


print("\nPart II finished.")
print(
    "Results were saved in the results folder."
)
