// reduction_kernel.cu
__global__ void reduction_kernel(unsigned int *hash_array, unsigned int *nonce_array, unsigned int *min_hash_array, unsigned int *min_nonce_array, unsigned int array_size)
{
    extern __shared__ unsigned int shared_data[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    unsigned int min_hash = UINT_MAX;
    unsigned int min_nonce = 0;

    // Reduce multiple elements per thread
    while (i < array_size)
    {
        if (hash_array[i] < min_hash)
        {
            min_hash = hash_array[i];
            min_nonce = nonce_array[i];
        }
        if (i + blockDim.x < array_size && hash_array[i + blockDim.x] < min_hash)
        {
            min_hash = hash_array[i + blockDim.x];
            min_nonce = nonce_array[i + blockDim.x];
        }
        i += gridSize;
    }

    // Each thread puts its local min into shared memory
    shared_data[tid] = min_hash;
    shared_data[tid + blockDim.x] = min_nonce;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (shared_data[tid + s] < shared_data[tid])
            {
                shared_data[tid] = shared_data[tid + s];
                shared_data[tid + blockDim.x] = shared_data[tid + blockDim.x + s];
            }
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0)
    {
        min_hash_array[blockIdx.x] = shared_data[0];
        min_nonce_array[blockIdx.x] = shared_data[blockDim.x];
    }
}
