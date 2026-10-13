#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

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

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (!A || !B || !C)
    {
        printf("CPU memory allocation failed\n");
        return 1;
    }

    srand(0);

    for (int i = 0; i < N * N; i++)
    {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);

    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    float elapsed = 0.0f;

    cudaEventElapsedTime(&elapsed, start, stop);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Tiled CUDA execution time (N=%d): %f ms\n", N, elapsed);
    printf("C[0] = %f\n", C[0]);

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
