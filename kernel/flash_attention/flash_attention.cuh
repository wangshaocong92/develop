#pragma once





namespace kernel {

/*
bc = [M/4d] M = block smem size
br = min(d,bc) 不大于d，主要是两方面原因。
    1. 切出来的q子矩阵的列为d，行大于d，qv后的秩最大也只会是d
    2. 
*/
template<int n,int d,int bc, int br>
__global__ void flash_attention_kernel(float* q, float* k, float* v, float* output,
                                        int batch_size, int seq_len,
                                        int head_dim) {
    // Kernel implementation goes here

}

}