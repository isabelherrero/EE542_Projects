#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>


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

    // Create cuBLAS handle
    cublasHandle_t handle;

    cublasStatus_t status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasCreate failed\n");
        return 1;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    // Warm-up call
    status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        d_B,
        N,
        d_A,
        N,
        &beta,
        d_C,
        N
    );

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cuBLAS warm-up failed\n");
        return 1;
    }

    cudaDeviceSynchronize();

    // CUDA events for timing
    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Matrix multiplication using cuBLAS
    status = cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        N,
        N,
        &alpha,
        d_B,
        N,
        d_A,
        N,
        &beta,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("cublasSgemm failed\n");
        return 1;
    }

    // Get elapsed GPU time
    float elapsed = 0.0f;

    cudaEventElapsedTime(&elapsed, start, stop);

    // Copy result back to CPU
    error = cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("Copy C to CPU failed: %s\n", cudaGetErrorString(error));
        return 1;
    }

    printf("cuBLAS execution time (N=%d): %f ms\n", N, elapsed);
    printf("C[0] = %f\n", C[0]);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

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