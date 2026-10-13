#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#define TILE_WIDTH 16


__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
    __shared__ float aTile[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float bTile[2][TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;

    int tiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;

    cuda::pipeline<cuda::thread_scope_thread> pipe =
        cuda::make_pipeline();

    // load first tile into buffer 0
    pipe.producer_acquire();

    int ac = tx;
    int br = ty;

    if (row < N && ac < N) {
        cuda::memcpy_async(
            &aTile[0][ty][tx],
            &A[row * N + ac],
            sizeof(float),
            pipe
        );
    } else {
        aTile[0][ty][tx] = 0.0f;
    }

    if (br < N && col < N) {
        cuda::memcpy_async(
            &bTile[0][ty][tx],
            &B[br * N + col],
            sizeof(float),
            pipe
        );
    } else {
        bTile[0][ty][tx] = 0.0f;
    }

    pipe.producer_commit();

    for (int t = 0; t < tiles; t++) {
        int cur = t % 2;
        int next = (t + 1) % 2;

        // wait for current tile
        pipe.consumer_wait();

        __syncthreads();

        // start loading next tile
        if (t + 1 < tiles) {
            int nextA =
                (t + 1) * TILE_WIDTH + tx;

            int nextB =
                (t + 1) * TILE_WIDTH + ty;

            pipe.producer_acquire();

            if (row < N && nextA < N) {
                cuda::memcpy_async(
                    &aTile[next][ty][tx],
                    &A[row * N + nextA],
                    sizeof(float),
                    pipe
                );
            } else {
                aTile[next][ty][tx] = 0.0f;
            }

            if (nextB < N && col < N) {
                cuda::memcpy_async(
                    &bTile[next][ty][tx],
                    &B[nextB * N + col],
                    sizeof(float),
                    pipe
                );
            } else {
                bTile[next][ty][tx] = 0.0f;
            }

            pipe.producer_commit();
        }

        // compute current tile
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum +=
                aTile[cur][ty][k] *
                bTile[cur][k][tx];
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
        "Double Buffer N=%d: %.6f ms\n",
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
