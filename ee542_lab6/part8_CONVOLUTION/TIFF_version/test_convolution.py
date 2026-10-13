import ctypes
import sys
import time

import numpy as np
from PIL import Image


def make_filter(name, n):
    if name == "blur":
        return np.full(
            (n, n),
            1.0 / (n * n),
            dtype=np.float32
        )

    if name == "edge":
        f = np.full(
            (n, n),
            -1.0,
            dtype=np.float32
        )
        f[n // 2, n // 2] = n * n - 1
        return f

    if name == "sharpen":
        f = np.full(
            (n, n),
            -1.0,
            dtype=np.float32
        )
        f[n // 2, n // 2] = n * n
        return f

    raise ValueError("Unknown filter")


if len(sys.argv) != 5:
    print(
        "Usage: python3 test_convolution.py "
        "input.tiff output.tiff filter N"
    )
    sys.exit(1)


in_file = sys.argv[1]
out_file = sys.argv[2]
name = sys.argv[3]
n = int(sys.argv[4])


img = Image.open(in_file).convert("L")
arr = np.array(img, dtype=np.uint8)

h, w = arr.shape

if w != h:
    raise ValueError("Image must be square")

m = w

out = np.zeros(
    (m, m),
    dtype=np.uint8
)

f = make_filter(name, n)


lib = ctypes.cdll.LoadLibrary(
    "./libconvolution.so"
)

u8_array = np.ctypeslib.ndpointer(
    dtype=np.uint8,
    ndim=1,
    flags="C_CONTIGUOUS"
)

f32_array = np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=1,
    flags="C_CONTIGUOUS"
)

lib.gpu_convolution.argtypes = [
    u8_array,
    u8_array,
    f32_array,
    ctypes.c_int,
    ctypes.c_int
]

lib.gpu_convolution.restype = ctypes.c_float


start = time.perf_counter()

gpu_ms = lib.gpu_convolution(
    arr.ravel(),
    out.ravel(),
    f.ravel(),
    m,
    n
)

end = time.perf_counter()


total_ms = (end - start) * 1000


print(f"Image: {m}x{m}")
print(f"Filter: {name} {n}x{n}")
print(f"GPU kernel time: {gpu_ms:.6f} ms")
print(f"Python library time: {total_ms:.6f} ms")


Image.fromarray(out).save(out_file)

print("Saved", out_file)
