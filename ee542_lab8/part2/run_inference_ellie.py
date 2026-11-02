
import torch
import tensorrt as trt
from pathlib import Path

MODEL_DIR = Path("models")
logger = trt.Logger(trt.Logger.WARNING)

def run_inference(precision, batch=1):
    engine_path = MODEL_DIR / f"resnet18_{precision}.engine"

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(
        engine_path.read_bytes()
    )

    if engine is None:
        raise RuntimeError("Could not load TensorRT engine")

    context = engine.create_execution_context()
    context.set_input_shape("input", (batch, 3, 224, 224))

    # Use the data types required by TensorRT
    input_dtype = engine.get_tensor_dtype("input")
    output_dtype = engine.get_tensor_dtype("logits")

    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
    }

    inp = torch.randn(
        batch, 3, 224, 224,
        device="cuda",
        dtype=dtype_map[input_dtype]
    )

    output_shape = tuple(context.get_tensor_shape("logits"))
    out = torch.empty(
        output_shape,
        device="cuda",
        dtype=dtype_map[output_dtype]
    )

    # Bind existing GPU memory buffers
    context.set_tensor_address("input", inp.data_ptr())
    context.set_tensor_address("logits", out.data_ptr())

    stream = torch.cuda.Stream()

    # Queue inference on the CUDA stream
    with torch.cuda.stream(stream):
        success = context.execute_async_v3(stream.cuda_stream)

    if not success:
        raise RuntimeError("TensorRT inference failed")

    # Wait until GPU inference finishes
    stream.synchronize()

    print(f"{precision.upper()} inference successful!")
    print("Input shape:", tuple(inp.shape))
    print("Output shape:", tuple(out.shape))
    print("Top-1 class:", out.argmax(dim=1).tolist())

    return out


if __name__ == "__main__":
    run_inference("fp32", batch=1)
    run_inference("fp16", batch=1)
