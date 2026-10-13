#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#define TILE_WIDTH 16


__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    __shared__ float aTile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bTile[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    cuda::pipeline<cuda::thread_scope_thread> pipe =
        cuda::make_pipeline();

    int tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < tiles; t++) {
        int ac = t * TILE_WIDTH + tx;
        int br = t * TILE_WIDTH + ty;

        pipe.producer_acquire();

        if (row < N && ac < N) {
            cuda::memcpy_async(
                &aTile[ty][tx],
                &A[row * N + ac],
                sizeof(float),
                pipe
            );
        } else {
            aTile[ty][tx] = 0.0f;
        }

        if (br < N && col < N) {
            cuda::memcpy_async(
                &bTile[ty][tx],
                &B[br * N + col],
                sizeof(float),
                pipe
            );
        } else {
            bTile[ty][tx] = 0.0f;
        }

        pipe.producer_commit();
        pipe.consumer_wait();

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += aTile[ty][k] * bTile[k][tx];
        }

        __syncthreads();

        pipe.consumer_release();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t size = (size_t)N * N * sizeof(float);

    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(
        d_A,
        A,
        size,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        B,
        size,
        cudaMemcpyHostToDevice
    );

    dim3 block(
        TILE_WIDTH,
        TILE_WIDTH
    );

    dim3 grid(
        (N + TILE_WIDTH - 1) / TILE_WIDTH,
        (N + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // warm-up
    matrixMultiply<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaDeviceSynchronize();

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMultiply<<<grid, block>>>(
        d_A,
        d_B,
        d_C,
        N
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;

    cudaEventElapsedTime(
        &ms,
        start,
        stop
    );

    cudaMemcpy(
        C,
        d_C,
        size,
        cudaMemcpyDeviceToHost
    );

    printf(
        "Async N=%d: %.6f ms\n",
        N,
        ms
    );

    printf(
        "C[0] = %.2f\n",
        C[0]
    );

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
