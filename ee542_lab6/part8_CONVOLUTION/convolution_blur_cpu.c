#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

void convolutionCPU(
    const unsigned char *image,
    unsigned char *output,
    const float *filter,
    int M,
    int N,
    int useAbsoluteValue
) {
    int radius = N / 2;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < M; col++) {
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
    }
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

    unsigned char *image = malloc(imageSize);
    unsigned char *output = malloc(imageSize);
    float *filter = malloc(filterSize);

    if (image == NULL || output == NULL || filter == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    generateImage(image, M);

    int useAbsoluteValue = 0;

    float value = 1.0f / (N * N);

        for (int i = 0; i < N * N; i++) {
            filter[i] = value;
        }

    struct timespec start;
    struct timespec end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    convolutionCPU(
        image,
        output,
        filter,
        M,
        N,
        useAbsoluteValue
    );

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed =
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf(
        "CPU convolution time (M=%d, N=%d): %.6f seconds\n",
        M,
        N,
        elapsed
    );

    char inputName[100];
    char outputName[100];

    snprintf(inputName, sizeof(inputName), "input_M%d.pgm", M);
    snprintf(
        outputName,
        sizeof(outputName),
        "cpu_blur_M%d_N%d.pgm",
        M,
        N
    );

    savePGM(inputName, image, M);
    savePGM(outputName, output, M);

    printf("Saved %s\n", inputName);
    printf("Saved %s\n", outputName);

    free(image);
    free(output);
    free(filter);

    return 0;
}
