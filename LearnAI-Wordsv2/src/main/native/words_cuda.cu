#include "words_cuda.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16

// --- CUDA GPU Kernels ---

extern "C" {

__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              int M, int N, int K,
                              int trans_a, int trans_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float val_a = trans_a ? a[k * M + row] : a[row * K + k];
            float val_b = trans_b ? b[col * K + k] : b[k * N + col];
            sum += val_a * val_b;
        }
        c[row * N + col] = sum;
    }
}

__global__ void matmul_shared_kernel(const float* a, const float* b, float* c,
                                     int M, int N, int K) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    for (int p = 0; p < (K + TILE_WIDTH - 1) / TILE_WIDTH; p++) {
        if (row < M && (p * TILE_WIDTH + tx) < K) {
            ds_A[ty][tx] = a[row * K + p * TILE_WIDTH + tx];
        } else {
            ds_A[ty][tx] = 0.0f;
        }
        
        if (col < N && (p * TILE_WIDTH + ty) < K) {
            ds_B[ty][tx] = b[(p * TILE_WIDTH + ty) * N + col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += ds_A[ty][k] * ds_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

__global__ void adam_update_kernel(float* w, const float* g, float* m, float* v,
                                   int size, float lr, float beta1, float beta2,
                                   float eps, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float bc1 = 1.0f - powf(beta1, t);
        float bc2 = 1.0f - powf(beta2, t);
        float oneMinusBeta1 = 1.0f - beta1;
        float oneMinusBeta2 = 1.0f - beta2;
        
        m[idx] = beta1 * m[idx] + oneMinusBeta1 * g[idx];
        v[idx] = beta2 * v[idx] + oneMinusBeta2 * g[idx] * g[idx];
        float mHat = m[idx] / bc1;
        float vHat = v[idx] / bc2;
        w[idx] -= lr * mHat / (sqrtf(vHat) + eps);
    }
}

__global__ void add_in_place_kernel(float* a, const float* b, int size, int a_cols, int b_size, int broadcast) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (broadcast == 1) {
            int col = idx % a_cols;
            a[idx] += b[col];
        } else if (broadcast == 2) {
            a[idx] += b[idx % b_size];
        } else {
            a[idx] += b[idx];
        }
    }
}

__global__ void subtract_in_place_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] -= b[idx];
    }
}

__global__ void multiply_element_wise_kernel(const float* a, const float* b, float* r, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        r[idx] = a[idx] * b[idx];
    }
}

__global__ void square_kernel(const float* a, float* r, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        r[idx] = a[idx] * a[idx];
    }
}

__global__ void sqrt_kernel(const float* a, float* r, int size, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        r[idx] = sqrtf(a[idx] + epsilon);
    }
}

__global__ void row_mean_kernel(const float* a, float* r, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        int offset = row * cols;
        for (int c = 0; c < cols; c++) {
            sum += a[offset + c];
        }
        r[row] = sum / cols;
    }
}

__global__ void row_variance_kernel(const float* a, const float* mean, float* r, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum_sq = 0.0f;
        float m = mean[row];
        int offset = row * cols;
        for (int c = 0; c < cols; c++) {
            float diff = a[offset + c] - m;
            sum_sq += diff * diff;
        }
        r[row] = sum_sq / cols;
    }
}

__global__ void transpose_kernel(const float* src, float* dest, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        dest[col * rows + row] = src[row * cols + col];
    }
}

__global__ void embedding_forward_kernel(const float* embeddings, const int* token_ids, float* output,
                                         int num_tokens, int embedding_dim) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens && dim_idx < embedding_dim) {
        int id = token_ids[token_idx];
        output[token_idx * embedding_dim + dim_idx] = embeddings[id * embedding_dim + dim_idx];
    }
}

__global__ void embedding_backward_kernel(const float* output_gradient, const int* token_ids, float* embeddings_gradient,
                                          int num_tokens, int embedding_dim) {
    int token_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx < num_tokens && dim_idx < embedding_dim) {
        int id = token_ids[token_idx];
        atomicAdd(&embeddings_gradient[id * embedding_dim + dim_idx], output_gradient[token_idx * embedding_dim + dim_idx]);
    }
}

__global__ void attention_forward_kernel(float* scores, int rows, int cols, float inv_scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int offset = row * cols;
        int activeLimit = (row % cols) + 1;
        
        float max_val = -1e20f;
        for (int j = 0; j < activeLimit; j++) {
            float val = scores[offset + j] * inv_scale;
            if (val > max_val) max_val = val;
        }
        
        float sum = 0.0f;
        for (int j = 0; j < activeLimit; j++) {
            float val = expf(scores[offset + j] * inv_scale - max_val);
            scores[offset + j] = val;
            sum += val;
        }
        
        for (int j = 0; j < activeLimit; j++) {
            float prob = scores[offset + j] / sum;
            if (prob < 1e-15f) prob = 1e-15f;
            if (prob > (1.0f - 1e-15f)) prob = 1.0f - 1e-15f;
            scores[offset + j] = prob;
        }
        
        for (int j = activeLimit; j < cols; j++) {
            scores[offset + j] = 1e-15f;
        }
    }
}

__global__ void attention_backward_kernel(const float* A, const float* dA, float* dS,
                                          int rows, int cols, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int offset = row * cols;
        int activeLimit = (row % cols) + 1;
        
        float dot = 0.0f;
        for (int k = 0; k < activeLimit; k++) {
            dot += dA[offset + k] * A[offset + k];
        }
        
        for (int j = 0; j < activeLimit; j++) {
            dS[offset + j] = A[offset + j] * (dA[offset + j] - dot) * scale;
        }
        for (int j = activeLimit; j < cols; j++) {
            dS[offset + j] = 0.0f;
        }
    }
}

__global__ void attention_q_k_forward_kernel(const float* q, const float* k, float* scores, int B, int T, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to B*T - 1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to T - 1
    if (row < B * T && col < T) {
        int b = row / T;
        float sum = 0.0f;
        int q_offset = row * d_model;
        int k_offset = (b * T + col) * d_model;
        for (int d = 0; d < d_model; d++) {
            sum += q[q_offset + d] * k[k_offset + d];
        }
        scores[row * T + col] = sum;
    }
}

__global__ void attention_out_forward_kernel(const float* scores, const float* v, float* output, int B, int T, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to B*T - 1
    int d = blockIdx.x * blockDim.x + threadIdx.x; // 0 to d_model - 1
    if (row < B * T && d < d_model) {
        int b = row / T;
        float sum = 0.0f;
        int s_offset = row * T;
        int v_offset = b * T * d_model + d;
        for (int j = 0; j < T; j++) {
            sum += scores[s_offset + j] * v[v_offset + j * d_model];
        }
        output[row * d_model + d] = sum;
    }
}

__global__ void attention_dv_backward_kernel(const float* A, const float* dO, float* dV, int B, int T, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to B*T - 1
    int d = blockIdx.x * blockDim.x + threadIdx.x; // 0 to d_model - 1
    if (row < B * T && d < d_model) {
        int b = row / T;
        int i = row % T;
        float sum = 0.0f;
        int dO_offset = b * T * d_model + d;
        for (int j = 0; j < T; j++) {
            sum += A[(b * T + j) * T + i] * dO[dO_offset + j * d_model];
        }
        dV[row * d_model + d] = sum;
    }
}

__global__ void attention_da_backward_kernel(const float* dO, const float* v, float* dA, int B, int T, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to B*T - 1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 0 to T - 1
    if (row < B * T && col < T) {
        int b = row / T;
        float sum = 0.0f;
        int do_offset = row * d_model;
        int v_offset = (b * T + col) * d_model;
        for (int d = 0; d < d_model; d++) {
            sum += dO[do_offset + d] * v[v_offset + d];
        }
        dA[row * T + col] = sum;
    }
}

__global__ void attention_dq_backward_kernel(const float* dS, const float* k, float* dQ, int B, int T, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to B*T - 1
    int d = blockIdx.x * blockDim.x + threadIdx.x; // 0 to d_model - 1
    if (row < B * T && d < d_model) {
        int b = row / T;
        float sum = 0.0f;
        int ds_offset = row * T;
        int k_offset = b * T * d_model + d;
        for (int j = 0; j < T; j++) {
            sum += dS[ds_offset + j] * k[k_offset + j * d_model];
        }
        dQ[row * d_model + d] = sum;
    }
}

__global__ void attention_dk_backward_kernel(const float* dS, const float* q, float* dK, int B, int T, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 0 to B*T - 1
    int d = blockIdx.x * blockDim.x + threadIdx.x; // 0 to d_model - 1
    if (row < B * T && d < d_model) {
        int b = row / T;
        int i = row % T;
        float sum = 0.0f;
        int q_offset = b * T * d_model + d;
        for (int j = 0; j < T; j++) {
            sum += dS[(b * T + j) * T + i] * q[q_offset + j * d_model];
        }
        dK[row * d_model + d] = sum;
    }
}

__global__ void softmax_forward_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int offset = row * cols;
        float max_val = -1e20f;
        for (int j = 0; j < cols; j++) {
            if (input[offset + j] > max_val) max_val = input[offset + j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float val = expf(input[offset + j] - max_val);
            output[offset + j] = val;
            sum += val;
        }
        
        for (int j = 0; j < cols; j++) {
            float prob = output[offset + j] / sum;
            if (prob < 1e-15f) prob = 1e-15f;
            if (prob > (1.0f - 1e-15f)) prob = 1.0f - 1e-15f;
            output[offset + j] = prob;
        }
    }
}

__global__ void layernorm_forward_kernel(const float* input, const float* gamma, const float* beta,
                                         float* output, float* x_hat, const float* mean, const float* var,
                                         int rows, int cols, float eps) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float m = mean[row];
        float v = var[row];
        float invStd = 1.0f / sqrtf(v + eps);
        int idx = row * cols + col;
        float xh = (input[idx] - m) * invStd;
        x_hat[idx] = xh;
        output[idx] = xh * gamma[col] + beta[col];
    }
}

__global__ void layernorm_backward_params_kernel(const float* output_gradient, const float* x_hat,
                                                 float* d_gamma, float* d_beta,
                                                 int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum_dg = 0.0f;
        float sum_db = 0.0f;
        for (int r = 0; r < rows; r++) {
            int idx = r * cols + col;
            sum_dg += output_gradient[idx] * x_hat[idx];
            sum_db += output_gradient[idx];
        }
        d_gamma[col] = sum_dg;
        d_beta[col] = sum_db;
    }
}

__global__ void layernorm_backward_input_kernel(const float* output_gradient, const float* x_hat, const float* var,
                                                const float* gamma, float* d_input,
                                                int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int offset = row * cols;
        float v = var[row];
        float invStd = 1.0f / sqrtf(v + eps);
        
        float dxh_sum = 0.0f;
        float xh_dxh_sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            int idx = offset + c;
            float dXHat = output_gradient[idx] * gamma[c];
            dxh_sum += dXHat;
            xh_dxh_sum += x_hat[idx] * dXHat;
        }
        float dxh_mean = dxh_sum / cols;
        float xh_dxh_mean = xh_dxh_sum / cols;
        
        for (int c = 0; c < cols; c++) {
            int idx = offset + c;
            float dXHat = output_gradient[idx] * gamma[c];
            d_input[idx] = invStd * (dXHat - dxh_mean - x_hat[idx] * xh_dxh_mean);
        }
    }
}

// --- Device Memory Wrappers ---

void* cuda_malloc(size_t size) {
    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    return dev_ptr;
}

void cuda_free(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

int cuda_memcpy_to_device(void* dest, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyHostToDevice failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_memcpy_to_host(void* dest, const void* src, size_t size) {
    cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpyDeviceToHost failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

// --- GPU Wrapper Launchers ---

static cublasHandle_t cublas_handle = NULL;

int cuda_matrix_multiply(const float* a, const float* b, float* c,
                         int M, int N, int K,
                         int trans_a, int trans_b) {
    if (cublas_handle == NULL) {
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed with status %d\n", status);
            return (int)status;
        }
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t transa_cublas = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb_cublas = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = trans_b ? K : N;
    int ldb = trans_a ? M : K;

    // Call cuBLAS SGEMM (handles all transpose combinations optimally using Tensor Cores)
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        transa_cublas,
        transb_cublas,
        N, M, K,
        &alpha,
        b, lda,
        a, ldb,
        &beta,
        c, N
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed with status %d\n", status);
        return (int)status;
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_matrix_multiply failed to synchronize: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_adam_update(float* w, const float* g, float* m, float* v,
                     int size, float lr, float beta1, float beta2,
                     float eps, int t) {
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    adam_update_kernel<<<numBlocks, threadsPerBlock>>>(w, g, m, v, size, lr, beta1, beta2, eps, t);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_adam_update failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_add_in_place(float* a, const float* b, int a_rows, int a_cols, int b_rows, int b_cols) {
    int size = a_rows * a_cols;
    int b_size = b_rows * b_cols;
    int broadcast = 0;
    if (b_rows == 1 && a_rows > 1) {
        broadcast = 1;
    } else if (b_rows > 1 && a_rows > b_rows) {
        broadcast = 2; // tiling
    }
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_in_place_kernel<<<numBlocks, threadsPerBlock>>>(a, b, size, a_cols, b_size, broadcast);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_add_in_place failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_subtract_in_place(float* a, const float* b, int size) {
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    subtract_in_place_kernel<<<numBlocks, threadsPerBlock>>>(a, b, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_subtract_in_place failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_multiply_element_wise(const float* a, const float* b, float* r, int size) {
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    multiply_element_wise_kernel<<<numBlocks, threadsPerBlock>>>(a, b, r, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_multiply_element_wise failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_square(const float* a, float* r, int size) {
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    square_kernel<<<numBlocks, threadsPerBlock>>>(a, r, size);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_square failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_sqrt(const float* a, float* r, int size, float epsilon) {
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    sqrt_kernel<<<numBlocks, threadsPerBlock>>>(a, r, size, epsilon);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_sqrt failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_row_mean(const float* a, float* r, int rows, int cols) {
    int threadsPerBlock = 256;
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
    row_mean_kernel<<<numBlocks, threadsPerBlock>>>(a, r, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_row_mean failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_row_variance(const float* a, const float* mean, float* r, int rows, int cols) {
    int threadsPerBlock = 256;
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
    row_variance_kernel<<<numBlocks, threadsPerBlock>>>(a, mean, r, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_row_variance failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_transpose(const float* src, float* dest, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    transpose_kernel<<<numBlocks, threadsPerBlock>>>(src, dest, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_transpose failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_embedding_forward(const float* embeddings, const int* token_ids, float* output,
                           int num_tokens, int embedding_dim) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((embedding_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_tokens + threadsPerBlock.y - 1) / threadsPerBlock.y);
    embedding_forward_kernel<<<numBlocks, threadsPerBlock>>>(embeddings, token_ids, output, num_tokens, embedding_dim);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_embedding_forward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_embedding_backward(const float* output_gradient, const int* token_ids, float* embeddings_gradient,
                            int num_tokens, int embedding_dim) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((embedding_dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (num_tokens + threadsPerBlock.y - 1) / threadsPerBlock.y);
    embedding_backward_kernel<<<numBlocks, threadsPerBlock>>>(output_gradient, token_ids, embeddings_gradient, num_tokens, embedding_dim);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_embedding_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_forward(float* scores, int rows, int cols, float inv_scale) {
    int threadsPerBlock = 256;
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
    attention_forward_kernel<<<numBlocks, threadsPerBlock>>>(scores, rows, cols, inv_scale);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_forward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_backward(const float* A, const float* dA, float* dS,
                            int rows, int cols, float scale) {
    int threadsPerBlock = 256;
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
    attention_backward_kernel<<<numBlocks, threadsPerBlock>>>(A, dA, dS, rows, cols, scale);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_q_k_forward(const float* q, const float* k, float* scores, int B, int T, int d_model) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((T + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B * T + threadsPerBlock.y - 1) / threadsPerBlock.y);
    attention_q_k_forward_kernel<<<numBlocks, threadsPerBlock>>>(q, k, scores, B, T, d_model);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_q_k_forward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_out_forward(const float* scores, const float* v, float* output, int B, int T, int d_model) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((d_model + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B * T + threadsPerBlock.y - 1) / threadsPerBlock.y);
    attention_out_forward_kernel<<<numBlocks, threadsPerBlock>>>(scores, v, output, B, T, d_model);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_out_forward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_dv_backward(const float* A, const float* dO, float* dV, int B, int T, int d_model) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((d_model + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B * T + threadsPerBlock.y - 1) / threadsPerBlock.y);
    attention_dv_backward_kernel<<<numBlocks, threadsPerBlock>>>(A, dO, dV, B, T, d_model);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_dv_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_da_backward(const float* dO, const float* v, float* dA, int B, int T, int d_model) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((T + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B * T + threadsPerBlock.y - 1) / threadsPerBlock.y);
    attention_da_backward_kernel<<<numBlocks, threadsPerBlock>>>(dO, v, dA, B, T, d_model);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_da_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_dq_backward(const float* dS, const float* k, float* dQ, int B, int T, int d_model) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((d_model + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B * T + threadsPerBlock.y - 1) / threadsPerBlock.y);
    attention_dq_backward_kernel<<<numBlocks, threadsPerBlock>>>(dS, k, dQ, B, T, d_model);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_dq_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_attention_dk_backward(const float* dS, const float* q, float* dK, int B, int T, int d_model) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((d_model + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (B * T + threadsPerBlock.y - 1) / threadsPerBlock.y);
    attention_dk_backward_kernel<<<numBlocks, threadsPerBlock>>>(dS, q, dK, B, T, d_model);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_attention_dk_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_softmax_forward(const float* input, float* output, int rows, int cols) {
    int threadsPerBlock = 256;
    int numBlocks = (rows + threadsPerBlock - 1) / threadsPerBlock;
    softmax_forward_kernel<<<numBlocks, threadsPerBlock>>>(input, output, rows, cols);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_softmax_forward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_layernorm_forward(const float* input, const float* gamma, const float* beta,
                           float* output, float* x_hat, const float* mean, const float* var,
                           int rows, int cols, float eps) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
    layernorm_forward_kernel<<<numBlocks, threadsPerBlock>>>(input, gamma, beta, output, x_hat, mean, var, rows, cols, eps);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_layernorm_forward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_layernorm_backward(const float* output_gradient, const float* x_hat, const float* var,
                            const float* gamma, float* d_input, float* d_gamma, float* d_beta,
                            int rows, int cols, float eps) {
    int threadsPerBlock = 256;
    int numBlocksCols = (cols + threadsPerBlock - 1) / threadsPerBlock;
    layernorm_backward_params_kernel<<<numBlocksCols, threadsPerBlock>>>(output_gradient, x_hat, d_gamma, d_beta, rows, cols);

    int numBlocksRows = (rows + threadsPerBlock - 1) / threadsPerBlock;
    layernorm_backward_input_kernel<<<numBlocksRows, threadsPerBlock>>>(output_gradient, x_hat, var, gamma, d_input, rows, cols, eps);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_layernorm_backward failed: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

__global__ void softmax_backward_and_loss_kernel(const float* probs, const int* targets, float* grads, float* loss_elements, int seq_len, int vocab_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = seq_len * vocab_size;
    if (idx < total_size) {
        int seq_idx = idx / vocab_size;
        int vocab_idx = idx % vocab_size;
        int target = targets[seq_idx];
        
        float prob = probs[idx];
        float clamped_prob = prob;
        if (clamped_prob < 1e-12f) clamped_prob = 1e-12f;
        if (clamped_prob > 1.0f) clamped_prob = 1.0f;
        
        float target_val = (vocab_idx == target) ? 1.0f : 0.0f;
        grads[idx] = (prob - target_val) / seq_len;
        
        if (vocab_idx == target) {
            loss_elements[seq_idx] = -logf(clamped_prob);
        }
    }
}

__global__ void sum_loss_kernel(const float* loss_elements, float* loss_out, int seq_len) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < seq_len; i += blockDim.x) {
        sum += loss_elements[i];
    }
    cache[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *loss_out = cache[0] / seq_len;
    }
}

__global__ void compute_norm_kernel(const float* grads, float* dev_norm, int vocab_size) {
    __shared__ float cache[256];
    int tid = threadIdx.x;
    
    float sumSq = 0.0f;
    for (int j = tid; j < vocab_size; j += blockDim.x) {
        float val = grads[j];
        sumSq += val * val;
    }
    cache[tid] = sumSq;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *dev_norm = sqrtf(cache[0] / vocab_size);
    }
}

__global__ void apply_clip_scale_kernel(float* grads, const float* dev_norm, int seq_len, int vocab_size) {
    float norm = *dev_norm;
    if (norm > 1.0f) {
        float clipScale = 1.0f / norm;
        int total_size = seq_len * vocab_size;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_size) {
            grads[idx] *= clipScale;
        }
    }
}

int cuda_softmax_backward_loss_clip(const float* probs, const int* targets, float* grads, float* loss_out, int seq_len, int vocab_size) {
    float* dev_loss_elements = NULL;
    float* dev_loss_val = NULL;
    float* dev_norm = NULL;
    
    cudaError_t err;
    err = cudaMalloc((void**)&dev_loss_elements, seq_len * sizeof(float));
    if (err != cudaSuccess) return (int)err;
    err = cudaMalloc((void**)&dev_loss_val, sizeof(float));
    if (err != cudaSuccess) { cudaFree(dev_loss_elements); return (int)err; }
    err = cudaMalloc((void**)&dev_norm, sizeof(float));
    if (err != cudaSuccess) { cudaFree(dev_loss_elements); cudaFree(dev_loss_val); return (int)err; }
    
    int total_size = seq_len * vocab_size;
    int threadsPerBlock = 256;
    int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;
    
    softmax_backward_and_loss_kernel<<<numBlocks, threadsPerBlock>>>(probs, targets, grads, dev_loss_elements, seq_len, vocab_size);
    
    sum_loss_kernel<<<1, 256>>>(dev_loss_elements, dev_loss_val, seq_len);
    
    compute_norm_kernel<<<1, 256>>>(grads, dev_norm, vocab_size);
    
    apply_clip_scale_kernel<<<numBlocks, threadsPerBlock>>>(grads, dev_norm, seq_len, vocab_size);
    
    err = cudaMemcpy(loss_out, dev_loss_val, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_loss_elements);
    cudaFree(dev_loss_val);
    cudaFree(dev_norm);
    
    if (err != cudaSuccess) return (int)err;
    return 0;
}

} // extern "C"
