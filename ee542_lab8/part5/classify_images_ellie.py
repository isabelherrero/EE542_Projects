
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tensorrt as trt

IMAGE_DIR = Path("results/part3")
ENGINE_PATH = Path("models/resnet18_fp16.engine")
RESULTS_DIR = Path("results/part5")

NUM_IMAGES = 16
HEIGHT, WIDTH, CHANNELS = 224, 224, 3
FRAME_BYTES = HEIGHT * WIDTH * CHANNELS

logger = trt.Logger(trt.Logger.WARNING)


def load_images(folder):
    frames = []
    for i in range(NUM_IMAGES):
        path = folder / f"frame_{i:02d}.bin"
        data = path.read_bytes()

        if len(data) != FRAME_BYTES:
            raise ValueError(f"Invalid image size: {path}")

        image = np.frombuffer(data, dtype=np.uint8)
        frames.append(image.reshape(HEIGHT, WIDTH, CHANNELS))

    return np.stack(frames).copy()


def copy_to_gpu(images):
    # Allocate page-locked CPU memory
    cpu_tensor = torch.from_numpy(images).pin_memory()

    gpu_tensor = torch.empty(
        cpu_tensor.shape,
        dtype=torch.uint8,
        device="cuda"
    )

    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm-up the transfer
    with torch.cuda.stream(stream):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    stream.synchronize()

    # Measure pinned CPU-to-GPU transfer
    with torch.cuda.stream(stream):
        start.record(stream)
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        end.record(stream)

    end.synchronize()
    copy_ms = start.elapsed_time(end)

    return gpu_tensor, copy_ms


def preprocess_on_gpu(images):
    # NHWC uint8 -> NCHW float32
    x = images.permute(0, 3, 1, 2).float() / 255.0

    # ResNet-18 pretrained weight transformations:
    # resize shorter edge to 256, center crop to 224
    x = F.interpolate(
        x,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
        antialias=True
    )

    x = x[:, :, 16:240, 16:240]

    mean = torch.tensor(
        [0.485, 0.456, 0.406],
        device="cuda"
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225],
        device="cuda"
    ).view(1, 3, 1, 1)

    x = (x - mean) / std

    return x.half().contiguous()


def classify(images):
    gpu_images, copy_ms = copy_to_gpu(images)
    inp = preprocess_on_gpu(gpu_images)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(
        ENGINE_PATH.read_bytes()
    )

    if engine is None:
        raise RuntimeError("Cannot load TensorRT engine")

    context = engine.create_execution_context()
    context.set_input_shape("input", tuple(inp.shape))

    output_shape = tuple(
        context.get_tensor_shape("logits")
    )

    out = torch.empty(
        output_shape,
        dtype=torch.float16,
        device="cuda"
    )

    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        # Ensure preprocessing finishes before inference
        stream.wait_stream(torch.cuda.current_stream())
        success = context.execute_async_v3(stream.cuda_stream)

    if not success:
        raise RuntimeError("TensorRT inference failed")

    stream.synchronize()

    predictions = out.argmax(dim=1).cpu().tolist()
    return predictions, copy_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--received-dir",
        type=Path,
        default=None,
        help="Directory containing reconstructed frame_XX.bin files"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_images = load_images(IMAGE_DIR)
    original_predictions, original_copy_ms = classify(
        original_images
    )

    weights = torchvision.models.ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]

    print("\n=== Original Image Classification ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Images: {NUM_IMAGES}")
    print(f"Pinned CPU-to-GPU copy: {original_copy_ms:.4f} ms")

    rows = []

    for i, prediction in enumerate(original_predictions):
        name = categories[prediction]
        print(f"Frame {i:02d}: Class {prediction} ({name})")
        rows.append({
            "Frame": i,
            "Original class": prediction,
            "Original label": name,
        })

    if args.received_dir:
        received_images = load_images(args.received_dir)
        received_predictions, received_copy_ms = classify(
            received_images
        )

        byte_matches = [
            np.array_equal(original_images[i], received_images[i])
            for i in range(NUM_IMAGES)
        ]

        class_matches = [
            original_predictions[i] == received_predictions[i]
            for i in range(NUM_IMAGES)
        ]

        for i, row in enumerate(rows):
            row["Received class"] = received_predictions[i]
            row["Byte match"] = byte_matches[i]
            row["Top-1 match"] = class_matches[i]

        print("\n=== Reconstruction Verification ===")
        print(f"Byte-identical frames: {sum(byte_matches)}/16")
        print(f"Top-1 agreement: {sum(class_matches)}/16")
        print(f"Received copy time: {received_copy_ms:.4f} ms")

    output_path = RESULTS_DIR / "classification_results.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
