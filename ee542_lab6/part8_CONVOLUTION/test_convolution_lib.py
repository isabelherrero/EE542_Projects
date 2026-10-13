#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        int aCol = m * TILE_WIDTH + threadIdx.x;
        int bRow = m * TILE_WIDTH + threadIdx.y;

        if (row < N && aCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum +=
                tileA[threadIdx.y][k] *
                tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void gpu_matrix_multiply(
    float *h_A,
    float *h_B,
    float *h_C,
    int N
) {
    size_t size = (size_t)N * N * sizeof(float);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void convolutionLibraryKernel(
    const unsigned char *image,
    unsigned char *output,
    const float *filter,
    int M,
    int N,
    int useAbsoluteValue
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= M) {
        return;
    }

    int radius = N / 2;
    float sum = 0.0f;

    for (int filterRow = 0; filterRow < N; filterRow++) {
        for (int filterCol = 0; filterCol < N; filterCol++) {
            int imageRow = row + filterRow - radius;
            int imageCol = col + filterCol - radius;

            if (
                imageRow >= 0 &&
                imageRow < M &&
                imageCol >= 0 &&
                imageCol < M
            ) {
                sum +=
                    image[imageRow * M + imageCol] *
                    filter[filterRow * N + filterCol];
            }
        }
    }

    if (useAbsoluteValue) {
        sum = fabsf(sum);
    }

    if (sum < 0.0f) {
        sum = 0.0f;
    }

    if (sum > 255.0f) {
        sum = 255.0f;
    }

    output[row * M + col] = (unsigned char)sum;
}

extern "C" void gpu_convolution(
    unsigned char *h_image,
    unsigned char *h_output,
    float *h_filter,
    int M,
    int N,
    int useAbsoluteValue
) {
    size_t imageSize =
        (size_t)M * M * sizeof(unsigned char);

    size_t filterSize =
        (size_t)N * N * sizeof(float);

    unsigned char *d_image;
    unsigned char *d_output;
    float *d_filter;

    cudaMalloc((void **)&d_image, imageSize);
    cudaMalloc((void **)&d_output, imageSize);
    cudaMalloc((void **)&d_filter, filterSize);

    cudaMemcpy(
        d_image,
        h_image,
        imageSize,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_filter,
        h_filter,
        filterSize,
        cudaMemcpyHostToDevice
    );

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);

    convolutionLibraryKernel<<<grid, block>>>(
        d_image,
        d_output,
        d_filter,
        M,
        N,
        useAbsoluteValue
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_output,
        d_output,
        imageSize,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_filter);
}
ubuntu@ip-172-31-32-52:~/lab6$ cat test_convolution_lib.py
import ctypes
import time
import numpy as np

lib = ctypes.cdll.LoadLibrary("./libmatrix.so")

uint8_array = np.ctypeslib.ndpointer(
    dtype=np.uint8,
    ndim=1,
    flags="C_CONTIGUOUS"
)

float32_array = np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=1,
    flags="C_CONTIGUOUS"
)

lib.gpu_convolution.argtypes = [
    uint8_array,
    uint8_array,
    float32_array,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

lib.gpu_convolution.restype = None

M = 1024
N = 7

row_values = (
    np.arange(M, dtype=np.int32) * 255 // M
).astype(np.uint8)

image = np.tile(row_values, (M, 1))

rows, cols = np.ogrid[:M, :M]
center = M // 2
radius = M // 4

circle = (
    (cols - center) ** 2 +
    (rows - center) ** 2
) < radius ** 2

image[circle] = 220

image[
    M // 8 + 1:M // 3,
    M // 8 + 1:M // 3
] = 40

output = np.zeros((M, M), dtype=np.uint8)

filter_matrix = np.full(
    (N, N),
    1.0 / (N * N),
    dtype=np.float32
)

start = time.perf_counter()

lib.gpu_convolution(
    image.ravel(),
    output.ravel(),
    filter_matrix.ravel(),
    M,
    N,
    0
)

end = time.perf_counter()

cold_time = end - start

output.fill(0)

start = time.perf_counter()

lib.gpu_convolution(
    image.ravel(),
    output.ravel(),
    filter_matrix.ravel(),
    M,
    N,
    0
)

end = time.perf_counter()

warm_time = end - start

filename = "python_gpu_output_M1024_N7.pgm"

with open(filename, "wb") as file:
    file.write(f"P5\n{M} {M}\n255\n".encode())
    file.write(output.tobytes())

print(f"First Python CUDA call: {cold_time:.6f} seconds")
print(f"Warm Python CUDA call: {warm_time:.6f} seconds")
print(f"Output shape: {output.shape}")
print(f"Saved {filename}")
