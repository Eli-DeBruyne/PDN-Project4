#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "support.h"

#define BLOCK_SIZE 16
#define FILTER_SIZE 5
#define FILTER_RADIUS (FILTER_SIZE / 2)

void writeMatrix(const char* filename, const int* matrix, int rows, int cols);
void readMatrix(const char* filename, int* matrix, int rows, int cols);

__global__ void convolutionKernel(const int *input, int *output, int rows, int cols, const int *filter) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int sum = 0;
        for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i) {
            for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j) {
                int r = row + i;
                int c = col + j;
                if (r >= 0 && r < rows && c >= 0 && c < cols) {
                    sum += input[r * cols + c] * filter[(i + FILTER_RADIUS) * FILTER_SIZE + (j + FILTER_RADIUS)];
                }
            }
        }
        output[row * cols + col] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s n_row n_col mat_input.csv mat_output_prob3.csv time_prob3_CUDA.csv\n", argv[0]);
        return EXIT_FAILURE;
    }

    int n_row = atoi(argv[1]);
    int n_col = atoi(argv[2]);
    const char *inputFile = argv[3];
    const char *outputFile = argv[4];
    const char *timingFile = argv[5];

    // Allocate memory for the host input matrix
    int *h_inputMatrix = (int *)malloc(sizeof(int) * n_row * n_col);
    int *h_outputMatrix = (int *)malloc(sizeof(int) * n_row * n_col);

    // Read the input matrix from file
    readMatrix(inputFile, h_inputMatrix, n_row, n_col);

    // Define the filter
    int h_filter[FILTER_SIZE * FILTER_SIZE] = {
        1, 0, 0, 0, 1,
        0, 1, 0, 1, 0,
        0, 0, 1, 0, 0,
        0, 1, 0, 1, 0,
        1, 0, 0, 0, 1
    };

    // Allocate memory for the device input, output, and filter
    int *d_inputMatrix, *d_outputMatrix, *d_filter;
    cudaMalloc(&d_inputMatrix, sizeof(int) * n_row * n_col);
    cudaMalloc(&d_outputMatrix, sizeof(int) * n_row * n_col);
    cudaMalloc(&d_filter, sizeof(int) * FILTER_SIZE * FILTER_SIZE);

    // Copy input data and filter from host to device
    cudaMemcpy(d_inputMatrix, h_inputMatrix, sizeof(int) * n_row * n_col, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(int) * FILTER_SIZE * FILTER_SIZE, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n_col + blockSize.x - 1) / blockSize.x, (n_row + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    Timer timer;
    startTime(&timer);
    convolutionKernel<<<gridSize, blockSize>>>(d_inputMatrix, d_outputMatrix, n_row, n_col, d_filter);
    cudaDeviceSynchronize();
    stopTime(&timer);

    // Copy the result back to host
    cudaMemcpy(h_outputMatrix, d_outputMatrix, sizeof(int) * n_row * n_col, cudaMemcpyDeviceToHost);

    // Write the result to file
    writeMatrix(outputFile, h_outputMatrix, n_row, n_col);

    // Record the execution time
    float totalTime = elapsedTime(timer);
    FILE *timeFile = fopen(timingFile, "w");
    if (timeFile == NULL) {
        fprintf(stderr, "Failed to open time file for writing\n");
        return EXIT_FAILURE;
    }
    fprintf(timeFile, "%f\n", totalTime);
    fclose(timeFile);

    // Free memory
    free(h_inputMatrix);
    free(h_outputMatrix);
    cudaFree(d_inputMatrix);
    cudaFree(d_outputMatrix);
    cudaFree(d_filter);

    return 0;
}

void readMatrix(const char* filename, int* matrix, int rows, int cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%d,", &matrix[i * cols + j]) != 1) {
                fprintf(stderr, "Error reading matrix from file\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void writeMatrix(const char* filename, const int* matrix, int rows, int cols) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%d", matrix[i * cols + j]);
            if (j != cols - 1) {
                fprintf(file, ",");
            }
        }
        if (i != rows - 1) {
            fprintf(file, "\n");
        }
    }
    fclose(file);
}
