#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>

// 支持最多3段组合
#define MAX_SEGMENTS 3

struct SegmentInfo {
    const char* values;
    int* offsets;
    int* lengths;
    int num_values;
};

__global__ void pcfg_generate_kernel(
    SegmentInfo* segs, int num_segments,
    int total, int max_guess_len, char* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int indices[MAX_SEGMENTS];
    int remain = idx;
    // 多重下标展开
    for (int i = num_segments - 1; i >= 0; --i) {
        indices[i] = remain % segs[i].num_values;
        remain /= segs[i].num_values;
    }
    char* out_ptr = output + idx * max_guess_len;
    int pos = 0;
    for (int i = 0; i < num_segments; ++i) {
        int off = segs[i].offsets[indices[i]];
        int len = segs[i].lengths[indices[i]];
        for (int j = 0; j < len && pos < max_guess_len - 1; ++j)
            out_ptr[pos++] = segs[i].values[off + j];
    }
    out_ptr[pos] = '\0';
}

extern "C" void PCFGGenerateAllGPU(
    const char*** all_values, // [num_segments][num_values]
    const int* num_values,    // 每段 value 数量
    int num_segments,
    int total,                // 组合总数
    int max_guess_len,
    char* output)
{
    SegmentInfo h_segs[MAX_SEGMENTS];
    char* d_values[MAX_SEGMENTS];
    int* d_offsets[MAX_SEGMENTS];
    int* d_lengths[MAX_SEGMENTS];
    for (int s = 0; s < num_segments; ++s) {
        int n = num_values[s];
        size_t all_len = 0;
        int* offsets = (int*)malloc(n * sizeof(int));
        int* lengths = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; ++i) {
            lengths[i] = strlen(all_values[s][i]);
            offsets[i] = all_len;
            all_len += lengths[i];
        }
        char* values = (char*)malloc(all_len);
        size_t pos = 0;
        for (int i = 0; i < n; ++i) {
            memcpy(values + pos, all_values[s][i], lengths[i]);
            pos += lengths[i];
        }
        cudaMalloc(&d_values[s], all_len);
        cudaMemcpy(d_values[s], values, all_len, cudaMemcpyHostToDevice);
        cudaMalloc(&d_offsets[s], n * sizeof(int));
        cudaMemcpy(d_offsets[s], offsets, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_lengths[s], n * sizeof(int));
        cudaMemcpy(d_lengths[s], lengths, n * sizeof(int), cudaMemcpyHostToDevice);
        h_segs[s].values = d_values[s];
        h_segs[s].offsets = d_offsets[s];
        h_segs[s].lengths = d_lengths[s];
        h_segs[s].num_values = n;
        free(offsets);
        free(lengths);
        free(values);
    }
    SegmentInfo* d_segs;
    cudaMalloc(&d_segs, num_segments * sizeof(SegmentInfo));
    cudaMemcpy(d_segs, h_segs, num_segments * sizeof(SegmentInfo), cudaMemcpyHostToDevice);
    char* d_output;
    cudaMalloc(&d_output, total * max_guess_len);
    int block = 128;
    int grid = (total + block - 1) / block;
    pcfg_generate_kernel<<<grid, block>>>(d_segs, num_segments, total, max_guess_len, d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, total * max_guess_len, cudaMemcpyDeviceToHost);
    for (int s = 0; s < num_segments; ++s) {
        cudaFree(d_values[s]);
        cudaFree(d_offsets[s]);
        cudaFree(d_lengths[s]);
    }
    cudaFree(d_segs);
    cudaFree(d_output);
}