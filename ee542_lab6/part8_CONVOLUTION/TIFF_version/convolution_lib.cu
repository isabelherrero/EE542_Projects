#include <cuda_runtime.h>


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

    if (sum < 0) {
        sum = 0;
    } else if (sum > 255) {
        sum = 255;
    }

    out[r * m + c] = (unsigned char)sum;
}


extern "C" float gpu_convolution(unsigned char *h_in,
                                 unsigned char *h_out,
                                 float *h_f,
                                 int m, int n) {
    size_t imgSize = (size_t)m * m;
    size_t fSize = (size_t)n * n * sizeof(float);

    unsigned char *d_in;
    unsigned char *d_out;
    float *d_f;

    cudaMalloc(&d_in, imgSize);
    cudaMalloc(&d_out, imgSize);
    cudaMalloc(&d_f, fSize);

    cudaMemcpy(d_in, h_in, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, fSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (m + block.x - 1) / block.x,
        (m + block.y - 1) / block.y
    );

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    conv<<<grid, block>>>(d_in, d_out, d_f, m, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, imgSize, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_f);

    return ms;
}
