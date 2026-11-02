
import csv
from pathlib import Path

import torch
import torchvision
import tensorrt as trt

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZES = [1, 8, 32]
WARMUP = 20
REPEATS = 100

torch.manual_seed(42)
torch.backends.cudnn.benchmark = False

logger = trt.Logger(trt.Logger.WARNING)

# Load pretrained PyTorch FP32 model
weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model = model.eval().cuda()


def load_trt(precision):
    path = MODEL_DIR / f"resnet18_{precision}.engine"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())

    if engine is None:
        raise RuntimeError(f"Failed to load {precision} engine")

    context = engine.create_execution_context()
    return runtime, engine, context


def prepare_trt(engine, context, x):
    batch = x.shape[0]
    context.set_input_shape("input", tuple(x.shape))

    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
    }

    required_dtype = dtype_map[engine.get_tensor_dtype("input")]
    inp = x.to(dtype=required_dtype).contiguous()

    output_shape = tuple(context.get_tensor_shape("logits"))
    output_dtype = dtype_map[engine.get_tensor_dtype("logits")]

    out = torch.empty(
        output_shape, device="cuda", dtype=output_dtype
    )

    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    return inp, out


def benchmark(fn, stream):
    # Warm-up
    with torch.cuda.stream(stream):
        for _ in range(WARMUP):
            fn()

    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []

    for _ in range(REPEATS):
        with torch.cuda.stream(stream):
            start.record(stream)
            fn()
            end.record(stream)

        end.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def main():
    rows = []

    trt_models = {
        "fp32": load_trt("fp32"),
        "fp16": load_trt("fp16"),
    }

    for batch in BATCH_SIZES:
        print(f"\n===== Batch {batch} =====")

        x = torch.randn(
            batch, 3, 224, 224,
            device="cuda", dtype=torch.float32
        )

        stream = torch.cuda.Stream()

        # PyTorch FP32 baseline
        with torch.inference_mode():
            with torch.cuda.stream(stream):
                ref = model(x)
            stream.synchronize()

            pt_time = benchmark(lambda: model(x), stream)

        rows.append({
            "Implementation": "PyTorch FP32",
            "Batch": batch,
            "Time (ms)": pt_time,
            "Throughput (images/s)": batch * 1000 / pt_time,
            "Speedup": 1.0,
            "Max difference": 0.0,
            "Top-1 agreement (%)": 100.0,
        })

        print(f"PyTorch FP32: {pt_time:.4f} ms")

        # TensorRT models
        for precision in ["fp32", "fp16"]:
            runtime, engine, context = trt_models[precision]

            inp, out = prepare_trt(engine, context, x)

            def run_trt():
                ok = context.execute_async_v3(stream.cuda_stream)
                if not ok:
                    raise RuntimeError("TensorRT execution failed")

            with torch.cuda.stream(stream):
                run_trt()
            stream.synchronize()

            trt_time = benchmark(run_trt, stream)

            max_diff = (
                ref.float() - out.float()
            ).abs().max().item()

            agreement = (
                ref.argmax(dim=1) == out.argmax(dim=1)
            ).float().mean().item() * 100

            name = f"TensorRT {precision.upper()}"

            rows.append({
                "Implementation": name,
                "Batch": batch,
                "Time (ms)": trt_time,
                "Throughput (images/s)": batch * 1000 / trt_time,
                "Speedup": pt_time / trt_time,
                "Max difference": max_diff,
                "Top-1 agreement (%)": agreement,
            })

            print(
                f"{name}: {trt_time:.4f} ms | "
                f"Speedup: {pt_time / trt_time:.2f}x | "
                f"Max diff: {max_diff:.6f} | "
                f"Top-1: {agreement:.1f}%"
            )

    output_file = RESULTS_DIR / "part2_results.csv"

    with output_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
