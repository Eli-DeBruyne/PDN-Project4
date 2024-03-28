// kernel.cu

#ifndef KERNEL_CU
#define KERNEL_CU

#include <cuda_runtime.h>

#define FILTER_SIZE 5
#define FILTER_RADIUS (FILTER_SIZE / 2)

// CUDA kernel for performing convolution
__global__ void convolution_kernel(int *input, int *output, int *filter, int n_row, int n_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_row && col < n_col) {
        int sum = 0;
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < n_row && c >= 0 && c < n_col) {
                    sum += input[r * n_col + c] * filter[(i + FILTER_RADIUS) * FILTER_SIZE + (j + FILTER_RADIUS)];
                }
            }
        }
        output[row * n_col + col] = sum;
    }
}

#endif
