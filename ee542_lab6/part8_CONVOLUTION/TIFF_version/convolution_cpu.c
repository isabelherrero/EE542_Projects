#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>


// read a 16-bit little endian value
uint16_t r16(FILE *f) {
    unsigned char b[2];
    fread(b, 1, 2, f);
    return b[0] | (b[1] << 8);
}


// read a 32-bit little endian value
uint32_t r32(FILE *f) {
    unsigned char b[4];
    fread(b, 1, 4, f);
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
}


// write a 16-bit little endian value
void w16(FILE *f, uint16_t x) {
    unsigned char b[2];

    b[0] = x & 255;
    b[1] = (x >> 8) & 255;

    fwrite(b, 1, 2, f);
}


// write a 32-bit little endian value
void w32(FILE *f, uint32_t x) {
    unsigned char b[4];

    b[0] = x & 255;
    b[1] = (x >> 8) & 255;
    b[2] = (x >> 16) & 255;
    b[3] = (x >> 24) & 255;

    fwrite(b, 1, 4, f);
}


// write one TIFF tag
void tag(FILE *f, int id, int type, int cnt, int val) {
    w16(f, id);
    w16(f, type);
    w32(f, cnt);
    w32(f, val);
}


// read a grayscale TIFF image
unsigned char *readTiff(char *name, int *w, int *h) {
    FILE *f = fopen(name, "rb");

    if (!f) {
        return NULL;
    }

    // get location of image file directory
    fseek(f, 4, SEEK_SET);
    uint32_t ifd = r32(f);

    fseek(f, ifd, SEEK_SET);
    int n = r16(f);

    uint32_t off = 0;
    uint32_t bytes = 0;

    // read TIFF tags
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

    // allocate space for image data
    unsigned char *img = malloc(bytes);

    // read image pixels
    fseek(f, off, SEEK_SET);
    fread(img, 1, bytes, f);

    fclose(f);

    return img;
}


// write a grayscale TIFF image
void writeTiff(char *name, unsigned char *img, int w, int h) {
    FILE *f = fopen(name, "wb");

    int n = 9;
    uint32_t ifd = 8;
    uint32_t off = ifd + 2 + n * 12 + 4;

    // TIFF little endian header
    fwrite("II", 1, 2, f);

    w16(f, 42);
    w32(f, ifd);
    w16(f, n);

    // write TIFF metadata
    tag(f, 256, 4, 1, w);
    tag(f, 257, 4, 1, h);
    tag(f, 258, 3, 1, 8);
    tag(f, 259, 3, 1, 1);
    tag(f, 262, 3, 1, 1);
    tag(f, 273, 4, 1, off);
    tag(f, 277, 3, 1, 1);
    tag(f, 278, 4, 1, h);
    tag(f, 279, 4, 1, w * h);

    // no next IFD
    w32(f, 0);

    // write image pixels
    fwrite(img, 1, w * h, f);

    fclose(f);
}


// create a blur filter
void blur(float *f, int n) {
    float v = 1.0f / (n * n);
    for (int i = 0; i < n * n; i++) {
        f[i] = v;
    }
}


// create an edge detection filter
void edge(float *f, int n) {
    for (int i = 0; i < n * n; i++) {
        f[i] = -1.0f;
    }
    f[(n * n) / 2] = n * n - 1;
}


// create a sharpening filter
void sharp(float *f, int n) {
    for (int i = 0; i < n * n; i++) {
        f[i] = -1.0f;
    }
    f[(n * n) / 2] = n * n;
}


// apply convolution to each pixel
void conv(unsigned char *in, unsigned char *out, float *f, int m, int n) {
    int d = n / 2;

    // loop through each image pixel
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < m; c++) {
            float sum = 0;

            // apply filter around current pixel
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int rr = r + i - d;
                    int cc = c + j - d;

                    // only use pixels inside the image
                    if (rr >= 0 && rr < m && cc >= 0 && cc < m) {
                        sum += in[rr * m + cc] * f[i * n + j];
                    }
                }
            }

            // keep pixel value in valid range
            if (sum < 0) {
                sum = 0;
            } else if (sum > 255) {
                sum = 255;
            }

            out[r * m + c] = (unsigned char)sum;
        }
    }
}


int main(int argc, char **argv) {
    // check command line arguments
    if (argc != 5) {
        printf("check ur args");
        return 1;
    }

    char *type = argv[3];
    int n = atoi(argv[4]);
    int w, h;

    // load input image
    unsigned char *in = readTiff(argv[1], &w, &h);

    if (!in) {
        printf("Could not open image\n");
        return 1;
    }

    int m = w;

    // allocate output image and filter
    unsigned char *out = malloc((size_t)m * m);
    float *f = malloc((size_t)n * n * sizeof(float));

    // build selected filter
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

    // time the convolution
    clock_t start = clock();

    conv(in, out, f, m, n);

    clock_t end = clock();

    double t = (double)(end - start) / CLOCKS_PER_SEC;

    // print performance results
    printf("Image: %dx%d\n", m, m);
    printf("Filter: %s %dx%d\n", type, n, n);
    printf("CPU time: %.6f seconds\n", t);

    // save processed image
    writeTiff(argv[2], out, m, m);

    // free allocated memory
    free(in);
    free(out);
    free(f);

    return 0;
}