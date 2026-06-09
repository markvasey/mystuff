package com.learnai.words.math;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

public class CudaBridge {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LOOKUP;

    private static final MethodHandle cudaMallocHandle;
    private static final MethodHandle cudaFreeHandle;
    private static final MethodHandle cudaMemcpyToDeviceHandle;
    private static final MethodHandle cudaMemcpyToHostHandle;
    private static final MethodHandle cudaMatrixMultiplyHandle;
    private static final MethodHandle cudaAdamUpdateHandle;
    
    // Milestone 2 Kernels
    private static final MethodHandle cudaAddInPlaceHandle;
    private static final MethodHandle cudaSubtractInPlaceHandle;
    private static final MethodHandle cudaMultiplyElementWiseHandle;
    private static final MethodHandle cudaSquareHandle;
    private static final MethodHandle cudaSqrtHandle;
    private static final MethodHandle cudaRowMeanHandle;
    private static final MethodHandle cudaRowVarianceHandle;
    private static final MethodHandle cudaTransposeHandle;

    // Milestone 4 Layers Kernels
    private static final MethodHandle cudaEmbeddingForwardHandle;
    private static final MethodHandle cudaEmbeddingBackwardHandle;
    private static final MethodHandle cudaAttentionForwardHandle;
    private static final MethodHandle cudaAttentionBackwardHandle;
    private static final MethodHandle cudaSoftmaxForwardHandle;
    private static final MethodHandle cudaLayerNormForwardHandle;
    private static final MethodHandle cudaLayerNormBackwardHandle;

    static {
        // Find the shared library libwords_cuda.so
        Path libPath = Path.of("libwords_cuda.so");
        if (!Files.exists(libPath)) {
            libPath = Path.of(System.getProperty("user.dir"), "libwords_cuda.so");
        }
        if (!Files.exists(libPath)) {
            libPath = Path.of(System.getProperty("user.dir"), "src", "main", "native", "libwords_cuda.so");
        }

        if (Files.exists(libPath)) {
            LOOKUP = SymbolLookup.libraryLookup(libPath.toAbsolutePath(), Arena.global());
        } else {
            throw new UnsatisfiedLinkError("Could not find libwords_cuda.so in working directory or target paths. " +
                    "Please run 'make -C src/main/native' to compile it.");
        }

        try {
            cudaMallocHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_malloc").orElseThrow(() -> new NoSuchMethodError("cuda_malloc")),
                FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            cudaFreeHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_free").orElseThrow(() -> new NoSuchMethodError("cuda_free")),
                FunctionDescriptor.ofVoid(ValueLayout.ADDRESS)
            );

            cudaMemcpyToDeviceHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_memcpy_to_device").orElseThrow(() -> new NoSuchMethodError("cuda_memcpy_to_device")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            cudaMemcpyToHostHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_memcpy_to_host").orElseThrow(() -> new NoSuchMethodError("cuda_memcpy_to_host")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG)
            );

            cudaMatrixMultiplyHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_matrix_multiply").orElseThrow(() -> new NoSuchMethodError("cuda_matrix_multiply")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // b
                    ValueLayout.ADDRESS, // c
                    ValueLayout.JAVA_INT, // M
                    ValueLayout.JAVA_INT, // N
                    ValueLayout.JAVA_INT, // K
                    ValueLayout.JAVA_INT, // trans_a
                    ValueLayout.JAVA_INT  // trans_b
                )
            );

            cudaAdamUpdateHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_adam_update").orElseThrow(() -> new NoSuchMethodError("cuda_adam_update")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // w
                    ValueLayout.ADDRESS, // g
                    ValueLayout.ADDRESS, // m
                    ValueLayout.ADDRESS, // v
                    ValueLayout.JAVA_INT, // size
                    ValueLayout.JAVA_FLOAT, // lr
                    ValueLayout.JAVA_FLOAT, // beta1
                    ValueLayout.JAVA_FLOAT, // beta2
                    ValueLayout.JAVA_FLOAT, // eps
                    ValueLayout.JAVA_INT   // t
                )
            );

            // Milestone 2 native bindings
            cudaAddInPlaceHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_add_in_place").orElseThrow(() -> new NoSuchMethodError("cuda_add_in_place")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // b
                    ValueLayout.JAVA_INT, // a_rows
                    ValueLayout.JAVA_INT, // a_cols
                    ValueLayout.JAVA_INT, // b_rows
                    ValueLayout.JAVA_INT  // b_cols
                )
            );

            cudaSubtractInPlaceHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_subtract_in_place").orElseThrow(() -> new NoSuchMethodError("cuda_subtract_in_place")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // b
                    ValueLayout.JAVA_INT  // size
                )
            );

            cudaMultiplyElementWiseHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_multiply_element_wise").orElseThrow(() -> new NoSuchMethodError("cuda_multiply_element_wise")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // b
                    ValueLayout.ADDRESS, // r
                    ValueLayout.JAVA_INT  // size
                )
            );

            cudaSquareHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_square").orElseThrow(() -> new NoSuchMethodError("cuda_square")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // r
                    ValueLayout.JAVA_INT  // size
                )
            );

            cudaSqrtHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_sqrt").orElseThrow(() -> new NoSuchMethodError("cuda_sqrt")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // r
                    ValueLayout.JAVA_INT, // size
                    ValueLayout.JAVA_FLOAT // epsilon
                )
            );

            cudaRowMeanHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_row_mean").orElseThrow(() -> new NoSuchMethodError("cuda_row_mean")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // r
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT  // cols
                )
            );

            cudaRowVarianceHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_row_variance").orElseThrow(() -> new NoSuchMethodError("cuda_row_variance")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // a
                    ValueLayout.ADDRESS, // mean
                    ValueLayout.ADDRESS, // r
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT  // cols
                )
            );

            cudaTransposeHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_transpose").orElseThrow(() -> new NoSuchMethodError("cuda_transpose")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // src
                    ValueLayout.ADDRESS, // dest
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT  // cols
                )
            );

            // Milestone 4 native bindings
            cudaEmbeddingForwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_embedding_forward").orElseThrow(() -> new NoSuchMethodError("cuda_embedding_forward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // embeddings
                    ValueLayout.ADDRESS, // token_ids
                    ValueLayout.ADDRESS, // output
                    ValueLayout.JAVA_INT, // num_tokens
                    ValueLayout.JAVA_INT  // embedding_dim
                )
            );

            cudaEmbeddingBackwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_embedding_backward").orElseThrow(() -> new NoSuchMethodError("cuda_embedding_backward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // output_gradient
                    ValueLayout.ADDRESS, // token_ids
                    ValueLayout.ADDRESS, // embeddings_gradient
                    ValueLayout.JAVA_INT, // num_tokens
                    ValueLayout.JAVA_INT  // embedding_dim
                )
            );

            cudaAttentionForwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_attention_forward").orElseThrow(() -> new NoSuchMethodError("cuda_attention_forward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // scores
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT, // cols
                    ValueLayout.JAVA_FLOAT // inv_scale
                )
            );

            cudaAttentionBackwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_attention_backward").orElseThrow(() -> new NoSuchMethodError("cuda_attention_backward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // A
                    ValueLayout.ADDRESS, // dA
                    ValueLayout.ADDRESS, // dS
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT, // cols
                    ValueLayout.JAVA_FLOAT // scale
                )
            );

            cudaSoftmaxForwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_softmax_forward").orElseThrow(() -> new NoSuchMethodError("cuda_softmax_forward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // input
                    ValueLayout.ADDRESS, // output
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT  // cols
                )
            );

            cudaLayerNormForwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_layernorm_forward").orElseThrow(() -> new NoSuchMethodError("cuda_layernorm_forward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // input
                    ValueLayout.ADDRESS, // gamma
                    ValueLayout.ADDRESS, // beta
                    ValueLayout.ADDRESS, // output
                    ValueLayout.ADDRESS, // x_hat
                    ValueLayout.ADDRESS, // mean
                    ValueLayout.ADDRESS, // var
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT, // cols
                    ValueLayout.JAVA_FLOAT // eps
                )
            );

            cudaLayerNormBackwardHandle = LINKER.downcallHandle(
                LOOKUP.find("cuda_layernorm_backward").orElseThrow(() -> new NoSuchMethodError("cuda_layernorm_backward")),
                FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, // output_gradient
                    ValueLayout.ADDRESS, // x_hat
                    ValueLayout.ADDRESS, // var
                    ValueLayout.ADDRESS, // gamma
                    ValueLayout.ADDRESS, // d_input
                    ValueLayout.ADDRESS, // dxhat_row_mean
                    ValueLayout.ADDRESS, // xhat_dxhat_row_mean
                    ValueLayout.JAVA_INT, // rows
                    ValueLayout.JAVA_INT, // cols
                    ValueLayout.JAVA_FLOAT // eps
                )
            );

        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize CudaBridge MethodHandles: " + e.getMessage(), e);
        }
    }

    public static MemorySegment cudaMalloc(long size) {
        try {
            MemorySegment ptr = (MemorySegment) cudaMallocHandle.invokeExact(size);
            if (ptr.equals(MemorySegment.NULL)) {
                throw new OutOfMemoryError("cudaMalloc returned NULL. VRAM allocation failed.");
            }
            return ptr;
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaMalloc: " + ex.getMessage(), ex);
        }
    }

    public static void cudaFree(MemorySegment ptr) {
        try {
            cudaFreeHandle.invokeExact(ptr);
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaFree: " + ex.getMessage(), ex);
        }
    }

    public static void cudaMemcpyToDevice(MemorySegment destDevicePtr, MemorySegment srcHostPtr, long size) {
        try {
            int status = (int) cudaMemcpyToDeviceHandle.invokeExact(destDevicePtr, srcHostPtr, size);
            if (status != 0) {
                throw new RuntimeException("cudaMemcpyToDevice failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaMemcpyToDevice: " + ex.getMessage(), ex);
        }
    }

    public static void cudaMemcpyToHost(MemorySegment destHostPtr, MemorySegment srcDevicePtr, long size) {
        try {
            int status = (int) cudaMemcpyToHostHandle.invokeExact(destHostPtr, srcDevicePtr, size);
            if (status != 0) {
                throw new RuntimeException("cudaMemcpyToHost failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaMemcpyToHost: " + ex.getMessage(), ex);
        }
    }

    public static void cudaMatrixMultiply(MemorySegment a, MemorySegment b, MemorySegment c,
                                          int M, int N, int K, boolean transA, boolean transB) {
        try {
            int tA = transA ? 1 : 0;
            int tB = transB ? 1 : 0;
            int status = (int) cudaMatrixMultiplyHandle.invokeExact(a, b, c, M, N, K, tA, tB);
            if (status != 0) {
                throw new RuntimeException("cudaMatrixMultiply failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaMatrixMultiply: " + ex.getMessage(), ex);
        }
    }

    public static void cudaAdamUpdate(MemorySegment w, MemorySegment g, MemorySegment m, MemorySegment v,
                                      int size, float lr, float beta1, float beta2, float eps, int t) {
        try {
            int status = (int) cudaAdamUpdateHandle.invokeExact(w, g, m, v, size, lr, beta1, beta2, eps, t);
            if (status != 0) {
                throw new RuntimeException("cudaAdamUpdate failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaAdamUpdate: " + ex.getMessage(), ex);
        }
    }

    public static void cudaAddInPlace(MemorySegment a, MemorySegment b, int aRows, int aCols, int bRows, int bCols) {
        try {
            int status = (int) cudaAddInPlaceHandle.invokeExact(a, b, aRows, aCols, bRows, bCols);
            if (status != 0) {
                throw new RuntimeException("cudaAddInPlace failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaAddInPlace: " + ex.getMessage(), ex);
        }
    }

    public static void cudaSubtractInPlace(MemorySegment a, MemorySegment b, int size) {
        try {
            int status = (int) cudaSubtractInPlaceHandle.invokeExact(a, b, size);
            if (status != 0) {
                throw new RuntimeException("cudaSubtractInPlace failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaSubtractInPlace: " + ex.getMessage(), ex);
        }
    }

    public static void cudaMultiplyElementWise(MemorySegment a, MemorySegment b, MemorySegment r, int size) {
        try {
            int status = (int) cudaMultiplyElementWiseHandle.invokeExact(a, b, r, size);
            if (status != 0) {
                throw new RuntimeException("cudaMultiplyElementWise failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaMultiplyElementWise: " + ex.getMessage(), ex);
        }
    }

    public static void cudaSquare(MemorySegment a, MemorySegment r, int size) {
        try {
            int status = (int) cudaSquareHandle.invokeExact(a, r, size);
            if (status != 0) {
                throw new RuntimeException("cudaSquare failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaSquare: " + ex.getMessage(), ex);
        }
    }

    public static void cudaSqrt(MemorySegment a, MemorySegment r, int size, float epsilon) {
        try {
            int status = (int) cudaSqrtHandle.invokeExact(a, r, size, epsilon);
            if (status != 0) {
                throw new RuntimeException("cudaSqrt failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaSqrt: " + ex.getMessage(), ex);
        }
    }

    public static void cudaRowMean(MemorySegment a, MemorySegment r, int rows, int cols) {
        try {
            int status = (int) cudaRowMeanHandle.invokeExact(a, r, rows, cols);
            if (status != 0) {
                throw new RuntimeException("cudaRowMean failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaRowMean: " + ex.getMessage(), ex);
        }
    }

    public static void cudaRowVariance(MemorySegment a, MemorySegment mean, MemorySegment r, int rows, int cols) {
        try {
            int status = (int) cudaRowVarianceHandle.invokeExact(a, mean, r, rows, cols);
            if (status != 0) {
                throw new RuntimeException("cudaRowVariance failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaRowVariance: " + ex.getMessage(), ex);
        }
    }

    public static void cudaTranspose(MemorySegment src, MemorySegment dest, int rows, int cols) {
        try {
            int status = (int) cudaTransposeHandle.invokeExact(src, dest, rows, cols);
            if (status != 0) {
                throw new RuntimeException("cudaTranspose failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaTranspose: " + ex.getMessage(), ex);
        }
    }

    public static void cudaEmbeddingForward(MemorySegment embeddings, MemorySegment tokenIds, MemorySegment output,
                                            int numTokens, int embeddingDim) {
        try {
            int status = (int) cudaEmbeddingForwardHandle.invokeExact(embeddings, tokenIds, output, numTokens, embeddingDim);
            if (status != 0) {
                throw new RuntimeException("cudaEmbeddingForward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaEmbeddingForward: " + ex.getMessage(), ex);
        }
    }

    public static void cudaEmbeddingBackward(MemorySegment outputGradient, MemorySegment tokenIds, MemorySegment embeddingsGradient,
                                             int numTokens, int embeddingDim) {
        try {
            int status = (int) cudaEmbeddingBackwardHandle.invokeExact(outputGradient, tokenIds, embeddingsGradient, numTokens, embeddingDim);
            if (status != 0) {
                throw new RuntimeException("cudaEmbeddingBackward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaEmbeddingBackward: " + ex.getMessage(), ex);
        }
    }

    public static void cudaAttentionForward(MemorySegment scores, int rows, int cols, float invScale) {
        try {
            int status = (int) cudaAttentionForwardHandle.invokeExact(scores, rows, cols, invScale);
            if (status != 0) {
                throw new RuntimeException("cudaAttentionForward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaAttentionForward: " + ex.getMessage(), ex);
        }
    }

    public static void cudaAttentionBackward(MemorySegment A, MemorySegment dA, MemorySegment dS,
                                             int rows, int cols, float scale) {
        try {
            int status = (int) cudaAttentionBackwardHandle.invokeExact(A, dA, dS, rows, cols, scale);
            if (status != 0) {
                throw new RuntimeException("cudaAttentionBackward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaAttentionBackward: " + ex.getMessage(), ex);
        }
    }

    public static void cudaSoftmaxForward(MemorySegment input, MemorySegment output, int rows, int cols) {
        try {
            int status = (int) cudaSoftmaxForwardHandle.invokeExact(input, output, rows, cols);
            if (status != 0) {
                throw new RuntimeException("cudaSoftmaxForward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaSoftmaxForward: " + ex.getMessage(), ex);
        }
    }

    public static void cudaLayerNormForward(MemorySegment input, MemorySegment gamma, MemorySegment beta,
                                            MemorySegment output, MemorySegment xHat, MemorySegment mean, MemorySegment var,
                                            int rows, int cols, float eps) {
        try {
            int status = (int) cudaLayerNormForwardHandle.invokeExact(input, gamma, beta, output, xHat, mean, var, rows, cols, eps);
            if (status != 0) {
                throw new RuntimeException("cudaLayerNormForward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaLayerNormForward: " + ex.getMessage(), ex);
        }
    }

    public static void cudaLayerNormBackward(MemorySegment outputGradient, MemorySegment xHat, MemorySegment var,
                                             MemorySegment gamma, MemorySegment dInput,
                                             MemorySegment dxhatRowMean, MemorySegment xhatDxhatRowMean,
                                             int rows, int cols, float eps) {
        try {
            int status = (int) cudaLayerNormBackwardHandle.invokeExact(outputGradient, xHat, var, gamma, dInput,
                    dxhatRowMean, xhatDxhatRowMean, rows, cols, eps);
            if (status != 0) {
                throw new RuntimeException("cudaLayerNormBackward failed with exit code: " + status);
            }
        } catch (Throwable ex) {
            throw new RuntimeException("Error executing cudaLayerNormBackward: " + ex.getMessage(), ex);
        }
    }
}
