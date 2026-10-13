#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define COARSEN_FACTOR 2
#define OUTPUT_TILE (TILE_WIDTH * COARSEN_FACTOR)

__global__ void matrixMultiplyAllOptimizations(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    __shared__ float tileA[OUTPUT_TILE][TILE_WIDTH + 1];
    __shared__ float tileB[TILE_WIDTH][OUTPUT_TILE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row0 = blockIdx.y * OUTPUT_TILE + ty;
    int row1 = row0 + TILE_WIDTH;

    int col0 = blockIdx.x * OUTPUT_TILE + tx;
    int col1 = col0 + TILE_WIDTH;

    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;

    int numberOfTiles =
        (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numberOfTiles; ++m)
    {
        int aCol = m * TILE_WIDTH + tx;
        int bRow = m * TILE_WIDTH + ty;

        tileA[ty][tx] =
            (row0 < N && aCol < N)
                ? A[row0 * N + aCol]
                : 0.0f;

        tileA[ty + TILE_WIDTH][tx] =
            (row1 < N && aCol < N)
                ? A[row1 * N + aCol]
                : 0.0f;

        tileB[ty][tx] =
            (bRow < N && col0 < N)
                ? B[bRow * N + col0]
                : 0.0f;

        tileB[ty][tx + TILE_WIDTH] =
            (bRow < N && col1 < N)
                ? B[bRow * N + col1]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            float a0 = tileA[ty][k];
            float a1 = tileA[ty + TILE_WIDTH][k];

            float b0 = tileB[k][tx];
            float b1 = tileB[k][tx + TILE_WIDTH];

            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < N && col0 < N)
        C[row0 * N + col0] = sum00;

    if (row0 < N && col1 < N)
        C[row0 * N + col1] = sum01;

    if (row1 < N && col0 < N)
        C[row1 * N + col0] = sum10;

    if (row1 < N && col1 < N)
        C[row1 * N + col1] = sum11;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    size_t elements = (size_t)N * N;
    size_t size = elements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (!h_A || !h_B || !h_C)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(0);

    for (size_t i = 0; i < elements; ++i)
    {
        h_A[i] = (rand() % 100) / 100.0f;
        h_B[i] = (rand() % 100) / 100.0f;
    }

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
        (N + OUTPUT_TILE - 1) / OUTPUT_TILE,
        (N + OUTPUT_TILE - 1) / OUTPUT_TILE
    );

    matrixMultiplyAllOptimizations<<<grid, block>>>(
        d_A, d_B, d_C, N
    );

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiplyAllOptimizations<<<grid, block>>>(
        d_A, d_B, d_C, N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf(
            "CUDA kernel error: %s\n",
            cudaGetErrorString(error)
        );
        return 1;
    }

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(
        h_C,
        d_C,
        size,
        cudaMemcpyDeviceToHost
    );

    printf(
        "All-optimizations CUDA kernel time (N=%d): %.6f ms\n",
        N,
        kernelTime
    );

    printf(
        "Check value: %f\n",
        h_C[0] + h_C[elements - 1]
    );

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
