#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

void generateImage(unsigned char *image, int M) {
    int center = M / 2;
    int radius = M / 4;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < M; col++) {
            int value = (col * 255) / M;

            int dx = col - center;
            int dy = row - center;

            if (dx * dx + dy * dy < radius * radius) {
                value = 220;
            }

            if (
                row > M / 8 &&
                row < M / 3 &&
                col > M / 8 &&
                col < M / 3
            ) {
                value = 40;
            }

            image[row * M + col] = (unsigned char)value;
        }
    }
}

__global__ void convolutionGPU(
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

void savePGM(
    const char *filename,
    const unsigned char *image,
    int M
) {
    FILE *file = fopen(filename, "wb");

    if (file == NULL) {
        printf("Could not create %s\n", filename);
        return;
    }

    fprintf(file, "P5\n%d %d\n255\n", M, M);
    fwrite(image, sizeof(unsigned char), M * M, file);
    fclose(file);
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 512;
    int N = (argc > 2) ? atoi(argv[2]) : 3;

    if (N % 2 == 0 || N < 3) {
        printf("Filter size must be an odd number of at least 3\n");
        return 1;
    }

    size_t imageSize = (size_t)M * M * sizeof(unsigned char);
    size_t filterSize = (size_t)N * N * sizeof(float);

    unsigned char *image = (unsigned char *)malloc(imageSize);
    unsigned char *output = (unsigned char *)malloc(imageSize);
    float *filter = (float *)malloc(filterSize);

    if (image == NULL || output == NULL || filter == NULL) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    generateImage(image, M);

    int useAbsoluteValue = 0;

    if (N == 3) {
        float edgeFilter[9] = {
            -1.0f, -1.0f, -1.0f,
            -1.0f,  8.0f, -1.0f,
            -1.0f, -1.0f, -1.0f
        };

        for (int i = 0; i < 9; i++) {
            filter[i] = edgeFilter[i];
        }

        useAbsoluteValue = 1;
    } else {
        float value = 1.0f / (N * N);

        for (int i = 0; i < N * N; i++) {
            filter[i] = value;
        }
    }

    unsigned char *d_image;
    unsigned char *d_output;
    float *d_filter;

    auto totalStart = std::chrono::high_resolution_clock::now();

    cudaMalloc((void **)&d_image, imageSize);
    cudaMalloc((void **)&d_output, imageSize);
    cudaMalloc((void **)&d_filter, filterSize);

    cudaMemcpy(
        d_image,
        image,
        imageSize,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_filter,
        filter,
        filterSize,
        cudaMemcpyHostToDevice
    );

    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (M + 15) / 16);

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    convolutionGPU<<<grid, block>>>(
        d_image,
        d_output,
        d_filter,
        M,
        N,
        useAbsoluteValue
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float kernelTime = 0.0f;
    cudaEventElapsedTime(&kernelTime, start, stop);

    cudaMemcpy(
        output,
        d_output,
        imageSize,
        cudaMemcpyDeviceToHost
    );

    auto totalEnd = std::chrono::high_resolution_clock::now();

    double totalTime =
        std::chrono::duration<double, std::milli>(
            totalEnd - totalStart
        ).count();

    printf(
        "GPU convolution kernel time (M=%d, N=%d): %f ms\n",
        M,
        N,
        kernelTime
    );

    printf(
        "GPU convolution total time (M=%d, N=%d): %f ms\n",
        M,
        N,
        totalTime
    );

    char outputName[100];

    snprintf(
        outputName,
        sizeof(outputName),
        "gpu_output_M%d_N%d.pgm",
        M,
        N
    );

    savePGM(outputName, output, M);
    printf("Saved %s\n", outputName);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_filter);

    free(image);
    free(output);
    free(filter);

    return 0;
}
