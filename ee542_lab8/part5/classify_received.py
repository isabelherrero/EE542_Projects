import csv
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
import torchvision


NUMBER_OF_IMAGES = 16
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
FRAME_BYTES = HEIGHT * WIDTH * CHANNELS
TOTAL_BYTES = NUMBER_OF_IMAGES * FRAME_BYTES

WARMUP_COPIES = 20
TIMED_COPIES = 100

LOGGER = trt.Logger(trt.Logger.WARNING)


def load_raw_images(filename):
    data = np.fromfile(filename, dtype=np.uint8)

    if data.size != TOTAL_BYTES:
        raise RuntimeError(
            f"{filename} contains {data.size} bytes; "
            f"expected {TOTAL_BYTES}"
        )

    return data.reshape(
        NUMBER_OF_IMAGES,
        HEIGHT,
        WIDTH,
        CHANNELS,
    )


def create_pinned_tensor(images):
    pinned = torch.empty(
        images.shape,
        dtype=torch.uint8,
        pin_memory=True,
    )
    pinned.copy_(torch.from_numpy(images))
    return pinned


def copy_to_gpu_and_measure(pinned):
    gpu_bytes = torch.empty(
        pinned.shape,
        dtype=torch.uint8,
        device="cuda",
    )

    stream = torch.cuda.Stream()

    for _ in range(WARMUP_COPIES):
        gpu_bytes.copy_(pinned, non_blocking=True)

    stream.synchronize()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream):
        start.record(stream)

        for _ in range(TIMED_COPIES):
            gpu_bytes.copy_(pinned, non_blocking=True)

        end.record(stream)

    end.synchronize()

    average_ms = start.elapsed_time(end) / TIMED_COPIES
    return gpu_bytes, average_ms


def copy_to_gpu(pinned):
    return pinned.to(device="cuda", non_blocking=True)


def preprocess_on_gpu(gpu_bytes):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    mean = torch.tensor(
        transform.mean,
        dtype=torch.float16,
        device="cuda",
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        transform.std,
        dtype=torch.float16,
        device="cuda",
    ).view(1, 3, 1, 1)

    images = gpu_bytes.permute(0, 3, 1, 2)
    images = images.to(dtype=torch.float16)
    images = images / 255.0
    images = (images - mean) / std

    return images.contiguous()


def load_engine(filename):
    runtime = trt.Runtime(LOGGER)

    with open(filename, "rb") as file:
        engine = runtime.deserialize_cuda_engine(file.read())

    if engine is None:
        raise RuntimeError(f"Could not load {filename}")

    return runtime, engine


def classify(engine, input_tensor):
    batch = input_tensor.shape[0]
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream()

    if not context.set_input_shape(
        "input",
        (batch, 3, HEIGHT, WIDTH),
    ):
        raise RuntimeError("Could not set TensorRT input shape")

    output = torch.empty(
        (batch, 1000),
        dtype=torch.float16,
        device="cuda",
    )

    if not context.set_tensor_address(
        "input",
        input_tensor.data_ptr(),
    ):
        raise RuntimeError("Could not set TensorRT input address")

    if not context.set_tensor_address(
        "logits",
        output.data_ptr(),
    ):
        raise RuntimeError("Could not set TensorRT output address")

    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT inference failed")

    stream.synchronize()
    return output


def main():
    original_file = "original_images.bin"
    received_file = "frames_rx_shuffled.bin"
    engine_file = "resnet18_fp16.engine"

    for filename in [original_file, received_file, engine_file]:
        if not Path(filename).exists():
            raise FileNotFoundError(filename)

    original_images = load_raw_images(original_file)
    received_images = load_raw_images(received_file)

    byte_match = np.array_equal(original_images, received_images)
    print(f"Received bytes match originals: {byte_match}")

    original_pinned = create_pinned_tensor(original_images)
    received_pinned = create_pinned_tensor(received_images)

    received_gpu, copy_ms = copy_to_gpu_and_measure(received_pinned)
    original_gpu = copy_to_gpu(original_pinned)

    torch.cuda.synchronize()

    original_input = preprocess_on_gpu(original_gpu)
    received_input = preprocess_on_gpu(received_gpu)

    runtime, engine = load_engine(engine_file)

    original_logits = classify(engine, original_input)
    received_logits = classify(engine, received_input)

    original_classes = original_logits.argmax(dim=1)
    received_classes = received_logits.argmax(dim=1)

    agreement = original_classes == received_classes
    agreement_percent = agreement.float().mean().item() * 100.0

    maximum_logit_difference = (
        original_logits.float() - received_logits.float()
    ).abs().max().item()

    categories = (
        torchvision.models.ResNet18_Weights.DEFAULT.meta["categories"]
    )

    copied_bytes = received_pinned.numel()
    bandwidth_gbps = (
        copied_bytes / (copy_ms / 1000.0) / 1_000_000_000
    )

    print(f"Pinned H2D copy size: {copied_bytes} bytes")
    print(f"Pinned H2D copy time: {copy_ms:.6f} ms")
    print(f"Pinned H2D bandwidth: {bandwidth_gbps:.3f} GB/s")
    print(f"Maximum logit difference: {maximum_logit_difference:.6f}")
    print(f"Top-1 agreement: {agreement_percent:.2f}%")
    print()

    rows = []

    print(
        f"{'Frame':<8}"
        f"{'Original':<12}"
        f"{'Received':<12}"
        f"{'Match':<8}"
        f"{'Predicted class'}"
    )

    for frame_id in range(NUMBER_OF_IMAGES):
        original_class = original_classes[frame_id].item()
        received_class = received_classes[frame_id].item()
        match = bool(agreement[frame_id].item())
        label = categories[received_class]

        print(
            f"{frame_id:<8}"
            f"{original_class:<12}"
            f"{received_class:<12}"
            f"{str(match):<8}"
            f"{label}"
        )

        rows.append([
            frame_id,
            original_class,
            received_class,
            match,
            label,
        ])

    with open("part5_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frame",
            "Original_class",
            "Received_class",
            "Match",
            "Predicted_label",
        ])
        writer.writerows(rows)

    with open("part5_summary.txt", "w") as file:
        file.write(f"Received bytes match originals: {byte_match}\n")
        file.write(f"Pinned H2D copy size: {copied_bytes} bytes\n")
        file.write(f"Pinned H2D copy time: {copy_ms:.6f} ms\n")
        file.write(
            f"Pinned H2D bandwidth: {bandwidth_gbps:.3f} GB/s\n"
        )
        file.write(
            f"Maximum logit difference: "
            f"{maximum_logit_difference:.6f}\n"
        )
        file.write(f"Top-1 agreement: {agreement_percent:.2f}%\n")

    print("\nCreated part5_results.csv")
    print("Created part5_summary.txt")

    del runtime


if __name__ == "__main__":
    main()