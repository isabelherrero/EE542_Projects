#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < N; k++)
        {
            sum += A[row * N + k] * B[k * N + col];
        }

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
        return 1;
    }

    error = cudaMalloc((void **)&d_C, size);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Copy matrices from CPU to GPU
    error = cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("Copy A to GPU failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    error = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("Copy B to GPU failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Configure CUDA thread blocks
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    // CUDA events for timing
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Run naïve CUDA matrix multiplication
    matrixMultiplyGPU<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Check for kernel launch errors
    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // Get elapsed GPU kernel time
    float elapsed = 0.0f;

    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy result back to CPU
    error = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("Copy C to CPU failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("Naive CUDA execution time (N=%d): %f ms\n", N, elapsed);

    // Print one output value so the compiler/program cannot ignore the result
    printf("C[0] = %f\n", C[0]);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    return 0;
}
