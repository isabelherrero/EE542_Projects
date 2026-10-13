#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 16
#define OUTPUT_TILE 32

__global__ void matrixMultiplyRegisterBlocked(
    const float *A,
    const float *B,
    float *C,
    int N
) {
    __shared__ float tileA[OUTPUT_TILE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][OUTPUT_TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row1 = blockIdx.y * OUTPUT_TILE + ty;
    int row2 = row1 + BLOCK_SIZE;

    int col1 = blockIdx.x * OUTPUT_TILE + tx;
    int col2 = col1 + BLOCK_SIZE;

    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum21 = 0.0f;
    float sum22 = 0.0f;

    int numberOfTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int m = 0; m < numberOfTiles; m++) {
        int aCol = m * BLOCK_SIZE + tx;
        int bRow = m * BLOCK_SIZE + ty;

        if (row1 < N && aCol < N) {
            tileA[ty][tx] = A[row1 * N + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (row2 < N && aCol < N) {
            tileA[ty + BLOCK_SIZE][tx] = A[row2 * N + aCol];
        } else {
            tileA[ty + BLOCK_SIZE][tx] = 0.0f;
        }

        if (bRow < N && col1 < N) {
            tileB[ty][tx] = B[bRow * N + col1];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        if (bRow < N && col2 < N) {
            tileB[ty][tx + BLOCK_SIZE] = B[bRow * N + col2];
        } else {
            tileB[ty][tx + BLOCK_SIZE] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a1 = tileA[ty][k];
            float a2 = tileA[ty + BLOCK_SIZE][k];
            float b1 = tileB[k][tx];
            float b2 = tileB[k][tx + BLOCK_SIZE];

            sum11 += a1 * b1;
            sum12 += a1 * b2;
            sum21 += a2 * b1;
            sum22 += a2 * b2;
        }

        __syncthreads();
    }

    if (row1 < N && col1 < N) {
        C[row1 * N + col1] = sum11;
    }

    if (row1 < N && col2 < N) {
        C[row1 * N + col2] = sum12;
    }

    if (row2 < N && col1 < N) {
        C[row2 * N + col1] = sum21;
    }

    if (row2 < N && col2 < N) {
        C[row2 * N + col2] = sum22;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    size_t size = (size_t)N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    if (A == NULL || B == NULL || C == NULL) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    srand(0);

    for (int i = 0; i < N * N; i++) {
        A[i] = (rand() % 100) / 100.0f;
        B[i] = (rand() % 100) / 100.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;

    auto totalStart = std::chrono::high_resolution_clock::now();

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + OUTPUT_TILE - 1) / OUTPUT_TILE,
        (N + OUTPUT_TILE - 1) / OUTPUT_TILE
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    matrixMultiplyRegisterBlocked<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);

    matrixMultiplyRegisterBlocked<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    auto totalEnd = std::chrono::high_resolution_clock::now();

    double totalTime =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    printf(
        "Register-blocked CUDA kernel time (N=%d): %f ms\n",
        N,
        kernelTime
    );

    printf(
        "Register-blocked CUDA total time (N=%d): %f ms\n",
        N,
        totalTime
    );

    printf("Check value: %f\n", C[0] + C[N * N - 1]);

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
