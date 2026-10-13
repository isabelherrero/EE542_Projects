#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMultiplyPadded(float *A, float *B, float *C, int N)
{
    // Extra column added to reduce potential shared-memory bank conflicts
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    int tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int m = 0; m < tiles; m++)
    {
        int aCol = m * TILE_WIDTH + tx;
        int bRow = m * TILE_WIDTH + ty;

        if (row < N && aCol < N)
        {
            ds_A[ty][tx] = A[row * N + aCol];
        }
        else
        {
            ds_A[ty][tx] = 0.0f;
        }

        if (bRow < N && col < N)
        {
            ds_B[ty][tx] = B[bRow * N + col];
        }
        else
        {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}


int main(int argc, char **argv)
{
    int N = (argc > 1) ? atoi(argv[1]) : 1024;

    size_t size = (size_t)N * N * sizeof(float);

    printf("Matrix size: %d x %d\n", N, N);

    // Allocate CPU memory
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (A == NULL || B == NULL || C == NULL)
    {
        printf("CPU memory allocation failed\n");
        return 1;
    }

    // Initialize matrices
    srand(0);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    // Allocate GPU memory
    float *d_A;
    float *d_B;
    float *d_C;

    cudaError_t error;

    error = cudaMalloc((void **)&d_A, size);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    error = cudaMalloc((void **)&d_B, size);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(error));
        cudaFree(d_A);
        return 1;
    }

    error = cudaMalloc((void **)&d_C, size);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(error));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }

    // Copy matrices to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);

    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Warm-up
    matrixMultiplyPadded<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // CUDA timing
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiplyPadded<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    float elapsed = 0.0f;

    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy result back
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Padded CUDA execution time (N=%d): %f ms\n", N, elapsed);
    printf("C[0] = %f\n", C[0]);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}