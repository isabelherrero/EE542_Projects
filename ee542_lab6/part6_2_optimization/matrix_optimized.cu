#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define COARSEN_FACTOR 2

__global__ void matrixMultiplyOptimized(
    float *A,
    float *B,
    float *C,
    int N)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH * COARSEN_FACTOR];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int colStart =
        blockIdx.x * (TILE_WIDTH * COARSEN_FACTOR) + tx;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < numTiles; ++m)
    {
        int aCol = m * TILE_WIDTH + tx;

        if (row < N && aCol < N)
            ds_A[ty][tx] = A[row * N + aCol];
        else
            ds_A[ty][tx] = 0.0f;

        int bRow = m * TILE_WIDTH + ty;

        int col0 = colStart;
        int col1 = colStart + TILE_WIDTH;

        if (bRow < N && col0 < N)
            ds_B[ty][tx] = B[bRow * N + col0];
        else
            ds_B[ty][tx] = 0.0f;

        if (bRow < N && col1 < N)
            ds_B[ty][tx + TILE_WIDTH] =
                B[bRow * N + col1];
        else
            ds_B[ty][tx + TILE_WIDTH] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            float a = ds_A[ty][k];

            sum0 += a * ds_B[k][tx];
            sum1 += a * ds_B[k][tx + TILE_WIDTH];
        }

        __syncthreads();
    }

    if (row < N && colStart < N)
        C[row * N + colStart] = sum0;

    if (row < N && colStart + TILE_WIDTH < N)
        C[row * N + colStart + TILE_WIDTH] = sum1;
}

int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    size_t size = (size_t)N * N * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    srand(0);

    for (int i = 0; i < N * N; i++)
    {
        h_A[i] = rand() % 100 / 100.0f;
        h_B[i] = rand() % 100 / 100.0f;
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
        (N + TILE_WIDTH * COARSEN_FACTOR - 1)
            / (TILE_WIDTH * COARSEN_FACTOR),
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    matrixMultiplyOptimized<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    matrixMultiplyOptimized<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;

    cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    );

    cudaMemcpy(
        h_C,
        d_C,
        size,
        cudaMemcpyDeviceToHost
    );

    printf(
        "Optimized CUDA execution time (N=%d): %.3f ms\n",
        N,
        milliseconds
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