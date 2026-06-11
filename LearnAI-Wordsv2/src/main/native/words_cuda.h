#ifndef WORDS_CUDA_H
#define WORDS_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device memory management
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
int cuda_memcpy_to_device(void* dest, const void* src, size_t size);
int cuda_memcpy_to_host(void* dest, const void* src, size_t size);

// Matrix Multiplication
int cuda_matrix_multiply(const float* a, const float* b, float* c,
                         int M, int N, int K,
                         int trans_a, int trans_b);

// Adam Optimizer Update
int cuda_adam_update(float* w, const float* g, float* m, float* v,
                     int size, float lr, float beta1, float beta2,
                     float eps, int t);

// Element-wise Operations
int cuda_add_in_place(float* a, const float* b, int a_rows, int a_cols, int b_rows, int b_cols);
int cuda_subtract_in_place(float* a, const float* b, int size);
int cuda_multiply_element_wise(const float* a, const float* b, float* r, int size);
int cuda_square(const float* a, float* r, int size);
int cuda_sqrt(const float* a, float* r, int size, float epsilon);

// Reductions
int cuda_row_mean(const float* a, float* r, int rows, int cols);
int cuda_row_variance(const float* a, const float* mean, float* r, int rows, int cols);

// Transposition
int cuda_transpose(const float* src, float* dest, int rows, int cols);

// Embedding Layer Operations
int cuda_embedding_forward(const float* embeddings, const int* token_ids, float* output,
                           int num_tokens, int embedding_dim);
int cuda_embedding_backward(const float* output_gradient, const int* token_ids, float* embeddings_gradient,
                            int num_tokens, int embedding_dim);

// Self Attention Custom Operations
int cuda_attention_forward(float* scores, int rows, int cols, float inv_scale);
int cuda_attention_backward(const float* A, const float* dA, float* dS,
                            int rows, int cols, float scale);
int cuda_attention_q_k_forward(const float* q, const float* k, float* scores, int B, int T, int d_model);
int cuda_attention_out_forward(const float* scores, const float* v, float* output, int B, int T, int d_model);
int cuda_attention_dv_backward(const float* A, const float* dO, float* dV, int B, int T, int d_model);
int cuda_attention_da_backward(const float* dO, const float* v, float* dA, int B, int T, int d_model);
int cuda_attention_dq_backward(const float* dS, const float* k, float* dQ, int B, int T, int d_model);
int cuda_attention_dk_backward(const float* dS, const float* q, float* dK, int B, int T, int d_model);

// Softmax Operations
int cuda_softmax_forward(const float* input, float* output, int rows, int cols);

// LayerNorm Operations
int cuda_layernorm_forward(const float* input, const float* gamma, const float* beta,
                           float* output, float* x_hat, const float* mean, const float* var,
                           int rows, int cols, float eps);
int cuda_layernorm_backward(const float* output_gradient, const float* x_hat, const float* var,
                            const float* gamma, float* d_input, float* d_gamma, float* d_beta,
                            int rows, int cols, float eps);

#ifdef __cplusplus
}
#endif

#endif // WORDS_CUDA_H
