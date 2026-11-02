
import torch
import torchvision
from pathlib import Path

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cuda")

# Load pretrained ResNet-18
weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights)
model = model.eval().to(device)

def export_model(model, precision, dtype):
    model = model.to(dtype=dtype)
    x = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)

    output_path = OUTPUT_DIR / f"resnet18_{precision}.onnx"

    torch.onnx.export(
        model,
        x,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"}
        },
        opset_version=18,
        dynamo=False
    )

    print(f"Exported: {output_path}")

export_model(model, "fp32", torch.float32)
export_model(model, "fp16", torch.float16)

print("ONNX export completed!")
