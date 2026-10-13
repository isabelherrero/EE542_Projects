#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>


uint16_t r16(FILE *f) {
    unsigned char b[2];
    fread(b, 1, 2, f);
    return b[0] | (b[1] << 8);
}


uint32_t r32(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);

    return b[0] |
           (b[1] << 8) |
           (b[2] << 16) |
           (b[3] << 24);
}


void w16(FILE *f, uint16_t x) {
    unsigned char b[2];

    b[0] = x & 255;
    b[1] = (x >> 8) & 255;

    fwrite(b, 1, 2, f);
}


void w32(FILE *f, uint32_t x) {
    unsigned char b[4];

    b[0] = x & 255;
    b[1] = (x >> 8) & 255;
    b[2] = (x >> 16) & 255;
    b[3] = (x >> 24) & 255;

    fwrite(b, 1, 4, f);
}


void tag(FILE *f, int id, int type, int cnt, int val) {
    w16(f, id);
    w16(f, type);
    w32(f, cnt);
    w32(f, val);
}


unsigned char *readTiff(char *name, int *w, int *h) {
    FILE *f = fopen(name, "rb");

    if (!f) {
        return NULL;
    }

    fseek(f, 4, SEEK_SET);
    uint32_t ifd = r32(f);

    fseek(f, ifd, SEEK_SET);
    int n = r16(f);

    uint32_t off = 0;
    uint32_t bytes = 0;

    for (int i = 0; i < n; i++) {
        int id = r16(f);

        r16(f);
        r32(f);

        uint32_t val = r32(f);

        if (id == 256) {
            *w = val;
        } else if (id == 257) {
            *h = val;
        } else if (id == 273) {
            off = val;
        } else if (id == 279) {
            bytes = val;
        }
    }

    unsigned char *img = (unsigned char *)malloc(bytes);

    fseek(f, off, SEEK_SET);
    fread(img, 1, bytes, f);

    fclose(f);

    return img;
}


void writeTiff(char *name, unsigned char *img, int w, int h) {
    FILE *f = fopen(name, "wb");

    int n = 9;
    uint32_t ifd = 8;
    uint32_t off = ifd + 2 + n * 12 + 4;

    fwrite("II", 1, 2, f);

    w16(f, 42);
    w32(f, ifd);
    w16(f, n);

    tag(f, 256, 4, 1, w);
    tag(f, 257, 4, 1, h);
    tag(f, 258, 3, 1, 8);
    tag(f, 259, 3, 1, 1);
    tag(f, 262, 3, 1, 1);
    tag(f, 273, 4, 1, off);
    tag(f, 277, 3, 1, 1);
    tag(f, 278, 4, 1, h);
    tag(f, 279, 4, 1, w * h);

    w32(f, 0);

    fwrite(img, 1, w * h, f);

    fclose(f);
}


void blur(float *f, int n) {
    float v = 1.0f / (n * n);

    for (int i = 0; i < n * n; i++) {
        f[i] = v;
    }
}


void edge(float *f, int n) {
    for (int i = 0; i < n * n; i++) {
        f[i] = -1.0f;
    }

    f[(n * n) / 2] = n * n - 1;
}


void sharp(float *f, int n) {
    for (int i = 0; i < n * n; i++) {
        f[i] = -1.0f;
    }

    f[(n * n) / 2] = n * n;
}


__global__ void conv(unsigned char *in, unsigned char *out,
                     float *f, int m, int n) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= m || c >= m) {
        return;
    }

    int d = n / 2;
    float sum = 0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int rr = r + i - d;
            int cc = c + j - d;

            if (rr >= 0 && rr < m && cc >= 0 && cc < m) {
                sum += in[rr * m + cc] * f[i * n + j];
            }
        }
    }
x
    if (sum < 0) {
        sum = 0;
    } else if (sum > 255) {
        sum = 255;
    }

    out[r * m + c] = (unsigned char)sum;
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s input.tiff output.tiff filter N\n", argv[0]);
        return 1;
    }

    char *type = argv[3];
    int n = atoi(argv[4]);

    int w, h;

    unsigned char *in = readTiff(argv[1], &w, &h);

    if (!in) {
        printf("Could not open image\n");
        return 1;
    }

    int m = w;
    size_t imgSize = (size_t)m * m;
    size_t fSize = (size_t)n * n * sizeof(float);

    unsigned char *out = (unsigned char *)malloc(imgSize);
    float *f = (float *)malloc(fSize);

    if (strcmp(type, "blur") == 0) {
        blur(f, n);
    } else if (strcmp(type, "edge") == 0) {
        edge(f, n);
    } else if (strcmp(type, "sharpen") == 0) {
        sharp(f, n);
    } else {
        printf("Unknown filter\n");
        return 1;
    }

    unsigned char *d_in;
    unsigned char *d_out;
    float *d_f;

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);
    cudaMalloc(&d_f, fSize);

    cudaMemcpy(d_in, in, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, fSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );

    // warm-up
    conv<<<grid, block>>>(d_in, d_out, d_f, m, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    conv<<<grid, block>>>(d_in, d_out, d_f, m, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t = 0;
    cudaEventElapsedTime(&t, start, stop);

    cudaMemcpy(out, d_out, imgSize, cudaMemcpyDeviceToHost);

    printf("Image: %dx%d\n", m, m);
    printf("Filter: %s %dx%d\n", type, n, n);
    printf("GPU time: %.6f ms\n", t);

    writeTiff(argv[2], out, m, m);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_f);

    free(in);
    free(out);
    free(f);

    return 0;
}
