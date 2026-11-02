
from pathlib import Path
import tensorrt as trt

MODEL_DIR = Path("models")

logger = trt.Logger(trt.Logger.WARNING)


def build_engine(precision):
    onnx_path = MODEL_DIR / f"resnet18_{precision}.onnx"
    engine_path = MODEL_DIR / f"resnet18_{precision}.engine"

    print(f"\nBuilding {precision.upper()} TensorRT engine...")

    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    # Load ONNX (including external weight files)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    config = builder.create_builder_config()

    # Dynamic batch optimization profile: 1 to 32
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input",
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    # Limit temporary build workspace to 1 GiB
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30
    )

    engine_data = builder.build_serialized_network(
        network, config
    )

    if engine_data is None:
        raise RuntimeError(f"Failed to build {precision} engine")

    engine_path.write_bytes(bytes(engine_data))

    print(f"Saved: {engine_path}")


if __name__ == "__main__":
    build_engine("fp32")
    build_engine("fp16")
    print("\nBoth TensorRT engines built successfully!")
