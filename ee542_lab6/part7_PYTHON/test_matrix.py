import ctypes
import os
import time
import numpy as np

# Load the CUDA DLL
dll_path = os.path.abspath("matrix_lib.dll")
lib = ctypes.CDLL(dll_path)

# Define argument types
lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="C_CONTIGUOUS"
    ),
    np.ctypeslib.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="C_CONTIGUOUS"
    ),
    np.ctypeslib.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="C_CONTIGUOUS"
    ),
    ctypes.c_int
]

lib.gpu_matrix_multiply.restype = None

N = 1024

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

start = time.time()

lib.gpu_matrix_multiply(
    A.ravel(),
    B.ravel(),
    C.ravel(),
    N
)

end = time.time()

print(
    f"Python call to CUDA library completed in {end - start:.4f} seconds"
)

# Correctness check
C_expected = A @ B

print(
    "Result correct:",
    np.allclose(C, C_expected, rtol=1e-3, atol=1e-3)
)