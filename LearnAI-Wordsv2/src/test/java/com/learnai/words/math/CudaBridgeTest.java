package com.learnai.words.math;

import org.junit.jupiter.api.Test;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import static org.junit.jupiter.api.Assertions.*;

public class CudaBridgeTest {

    @Test
    public void testCudaMemoryAllocationAndCopy() {
        long floatCount = 5;
        long byteSize = floatCount * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devPtr = CudaBridge.cudaMalloc(byteSize);
        assertNotNull(devPtr);
        assertNotEquals(MemorySegment.NULL, devPtr);

        try (Arena arena = Arena.ofConfined()) {
            float[] initialData = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
            MemorySegment hostSrc = arena.allocateFrom(ValueLayout.JAVA_FLOAT, initialData);

            CudaBridge.cudaMemcpyToDevice(devPtr, hostSrc, byteSize);

            MemorySegment hostDest = arena.allocate(ValueLayout.JAVA_FLOAT, floatCount);
            CudaBridge.cudaMemcpyToHost(hostDest, devPtr, byteSize);

            float[] resultData = hostDest.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(initialData, resultData, 1e-6f);
        } finally {
            CudaBridge.cudaFree(devPtr);
        }
    }

    @Test
    public void testCudaMatrixMultiply() {
        int M = 2;
        int K = 3;
        int N = 2;

        long sizeA = M * K * ValueLayout.JAVA_FLOAT.byteSize();
        long sizeB = K * N * ValueLayout.JAVA_FLOAT.byteSize();
        long sizeC = M * N * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(sizeA);
        MemorySegment devB = CudaBridge.cudaMalloc(sizeB);
        MemorySegment devC = CudaBridge.cudaMalloc(sizeC);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float[] dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostB = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataB);
            MemorySegment hostC = arena.allocate(ValueLayout.JAVA_FLOAT, M * N);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, sizeA);
            CudaBridge.cudaMemcpyToDevice(devB, hostB, sizeB);

            CudaBridge.cudaMatrixMultiply(devA, devB, devC, M, N, K, false, false);

            CudaBridge.cudaMemcpyToHost(hostC, devC, sizeC);

            float[] result = hostC.toArray(ValueLayout.JAVA_FLOAT);
            float[] expected = {58.0f, 64.0f, 139.0f, 154.0f};
            assertArrayEquals(expected, result, 1e-5f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devB);
            CudaBridge.cudaFree(devC);
        }
    }

    @Test
    public void testCudaAdamUpdate() {
        int size = 3;
        long byteSize = size * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devW = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devG = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devM = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devV = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] w = {1.0f, 2.0f, 3.0f};
            float[] g = {0.1f, -0.2f, 0.05f};
            float[] m = {0.01f, -0.02f, 0.005f};
            float[] v = {0.001f, 0.002f, 0.0005f};

            MemorySegment hostW = arena.allocateFrom(ValueLayout.JAVA_FLOAT, w);
            MemorySegment hostG = arena.allocateFrom(ValueLayout.JAVA_FLOAT, g);
            MemorySegment hostM = arena.allocateFrom(ValueLayout.JAVA_FLOAT, m);
            MemorySegment hostV = arena.allocateFrom(ValueLayout.JAVA_FLOAT, v);

            CudaBridge.cudaMemcpyToDevice(devW, hostW, byteSize);
            CudaBridge.cudaMemcpyToDevice(devG, hostG, byteSize);
            CudaBridge.cudaMemcpyToDevice(devM, hostM, byteSize);
            CudaBridge.cudaMemcpyToDevice(devV, hostV, byteSize);

            float lr = 0.001f;
            float beta1 = 0.9f;
            float beta2 = 0.999f;
            float eps = 1e-8f;
            int t = 1;

            CudaBridge.cudaAdamUpdate(devW, devG, devM, devV, size, lr, beta1, beta2, eps, t);

            MemorySegment hostWRes = arena.allocate(ValueLayout.JAVA_FLOAT, size);
            CudaBridge.cudaMemcpyToHost(hostWRes, devW, byteSize);

            float[] resultW = hostWRes.toArray(ValueLayout.JAVA_FLOAT);

            float expectedW0 = 1.0f - 0.001f * (0.19f) / ((float) Math.sqrt(1.009f) + 1e-8f);
            assertEquals(expectedW0, resultW[0], 1e-5f);
        } finally {
            CudaBridge.cudaFree(devW);
            CudaBridge.cudaFree(devG);
            CudaBridge.cudaFree(devM);
            CudaBridge.cudaFree(devV);
        }
    }

    @Test
    public void testCudaAddInPlace() {
        int r = 2;
        int c = 3;
        long byteSize = r * c * ValueLayout.JAVA_FLOAT.byteSize();
        long biasByteSize = 1 * c * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devB = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devBias = CudaBridge.cudaMalloc(biasByteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            float[] dataB = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
            float[] bias = {10.0f, 20.0f, 30.0f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostB = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataB);
            MemorySegment hostBias = arena.allocateFrom(ValueLayout.JAVA_FLOAT, bias);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);
            CudaBridge.cudaMemcpyToDevice(devB, hostB, byteSize);
            CudaBridge.cudaAddInPlace(devA, devB, r, c, r, c);
            CudaBridge.cudaMemcpyToHost(hostA, devA, byteSize);
            float[] res1 = hostA.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(new float[]{1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f}, res1, 1e-6f);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);
            CudaBridge.cudaMemcpyToDevice(devBias, hostBias, biasByteSize);
            CudaBridge.cudaAddInPlace(devA, devBias, r, c, 1, c);
            CudaBridge.cudaMemcpyToHost(hostA, devA, byteSize);
            float[] res2 = hostA.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(new float[]{11.5f, 22.5f, 33.5f, 14.5f, 25.5f, 36.5f}, res2, 1e-5f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devB);
            CudaBridge.cudaFree(devBias);
        }
    }

    @Test
    public void testCudaSubtractInPlace() {
        int size = 4;
        long byteSize = size * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devB = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {10.0f, 20.0f, 30.0f, 40.0f};
            float[] dataB = {1.0f, 2.0f, 3.0f, 4.0f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostB = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataB);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);
            CudaBridge.cudaMemcpyToDevice(devB, hostB, byteSize);

            CudaBridge.cudaSubtractInPlace(devA, devB, size);

            CudaBridge.cudaMemcpyToHost(hostA, devA, byteSize);
            float[] result = hostA.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(new float[]{9.0f, 18.0f, 27.0f, 36.0f}, result, 1e-6f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devB);
        }
    }

    @Test
    public void testCudaMultiplyElementWise() {
        int size = 3;
        long byteSize = size * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devB = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devR = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {2.0f, 3.0f, 4.0f};
            float[] dataB = {5.0f, 6.0f, 7.0f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostB = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataB);
            MemorySegment hostR = arena.allocate(ValueLayout.JAVA_FLOAT, size);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);
            CudaBridge.cudaMemcpyToDevice(devB, hostB, byteSize);

            CudaBridge.cudaMultiplyElementWise(devA, devB, devR, size);

            CudaBridge.cudaMemcpyToHost(hostR, devR, byteSize);
            float[] result = hostR.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(new float[]{10.0f, 18.0f, 28.0f}, result, 1e-6f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devB);
            CudaBridge.cudaFree(devR);
        }
    }

    @Test
    public void testCudaSquare() {
        int size = 3;
        long byteSize = size * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devR = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {2.0f, -3.0f, 5.0f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostR = arena.allocate(ValueLayout.JAVA_FLOAT, size);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);

            CudaBridge.cudaSquare(devA, devR, size);

            CudaBridge.cudaMemcpyToHost(hostR, devR, byteSize);
            float[] result = hostR.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(new float[]{4.0f, 9.0f, 25.0f}, result, 1e-6f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devR);
        }
    }

    @Test
    public void testCudaSqrt() {
        int size = 3;
        long byteSize = size * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devR = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {3.99f, 8.99f, 15.99f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostR = arena.allocate(ValueLayout.JAVA_FLOAT, size);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);

            CudaBridge.cudaSqrt(devA, devR, size, 0.01f);

            CudaBridge.cudaMemcpyToHost(hostR, devR, byteSize);
            float[] result = hostR.toArray(ValueLayout.JAVA_FLOAT);
            assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, result, 1e-5f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devR);
        }
    }

    @Test
    public void testCudaRowMeanAndVariance() {
        int rows = 2;
        int cols = 3;
        long byteSize = rows * cols * ValueLayout.JAVA_FLOAT.byteSize();
        long resByteSize = rows * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devMean = CudaBridge.cudaMalloc(resByteSize);
        MemorySegment devVar = CudaBridge.cudaMalloc(resByteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f};

            MemorySegment hostA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dataA);
            MemorySegment hostMean = arena.allocate(ValueLayout.JAVA_FLOAT, rows);
            MemorySegment hostVar = arena.allocate(ValueLayout.JAVA_FLOAT, rows);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);

            CudaBridge.cudaRowMean(devA, devMean, rows, cols);
            CudaBridge.cudaRowVariance(devA, devMean, devVar, rows, cols);

            CudaBridge.cudaMemcpyToHost(hostMean, devMean, resByteSize);
            CudaBridge.cudaMemcpyToHost(hostVar, devVar, resByteSize);

            float[] meanRes = hostMean.toArray(ValueLayout.JAVA_FLOAT);
            float[] varRes = hostVar.toArray(ValueLayout.JAVA_FLOAT);

            assertArrayEquals(new float[]{2.0f, 6.0f}, meanRes, 1e-6f);
            assertArrayEquals(new float[]{2.0f / 3.0f, 8.0f / 3.0f}, varRes, 1e-5f);
        } finally {
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devMean);
            CudaBridge.cudaFree(devVar);
        }
    }

    @Test
    public void testCudaTranspose() {
        int rows = 2;
        int cols = 3;
        long byteSize = rows * cols * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devSrc = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devDest = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] data = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            };

            MemorySegment hostSrc = arena.allocateFrom(ValueLayout.JAVA_FLOAT, data);
            MemorySegment hostDest = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);

            CudaBridge.cudaMemcpyToDevice(devSrc, hostSrc, byteSize);

            CudaBridge.cudaTranspose(devSrc, devDest, rows, cols);

            CudaBridge.cudaMemcpyToHost(hostDest, devDest, byteSize);
            float[] result = hostDest.toArray(ValueLayout.JAVA_FLOAT);

            float[] expected = {
                1.0f, 4.0f,
                2.0f, 5.0f,
                3.0f, 6.0f
            };
            assertArrayEquals(expected, result, 1e-6f);
        } finally {
            CudaBridge.cudaFree(devSrc);
            CudaBridge.cudaFree(devDest);
        }
    }

    @Test
    public void testCudaEmbedding() {
        int vocabSize = 4;
        int embDim = 3;
        int numTokens = 2;

        long embBytes = vocabSize * embDim * ValueLayout.JAVA_FLOAT.byteSize();
        long idsBytes = numTokens * ValueLayout.JAVA_INT.byteSize();
        long outBytes = numTokens * embDim * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devEmb = CudaBridge.cudaMalloc(embBytes);
        MemorySegment devIds = CudaBridge.cudaMalloc(idsBytes);
        MemorySegment devOut = CudaBridge.cudaMalloc(outBytes);

        try (Arena arena = Arena.ofConfined()) {
            float[] embeddings = {
                0.1f, 0.2f, 0.3f,
                1.1f, 1.2f, 1.3f,
                2.1f, 2.2f, 2.3f,
                3.1f, 3.2f, 3.3f
            };
            int[] tokenIds = {1, 3}; // Select embeddings index 1 and 3

            MemorySegment hostEmb = arena.allocateFrom(ValueLayout.JAVA_FLOAT, embeddings);
            MemorySegment hostIds = arena.allocateFrom(ValueLayout.JAVA_INT, tokenIds);
            MemorySegment hostOut = arena.allocate(ValueLayout.JAVA_FLOAT, numTokens * embDim);

            CudaBridge.cudaMemcpyToDevice(devEmb, hostEmb, embBytes);
            CudaBridge.cudaMemcpyToDevice(devIds, hostIds, idsBytes);

            // Forward
            CudaBridge.cudaEmbeddingForward(devEmb, devIds, devOut, numTokens, embDim);

            CudaBridge.cudaMemcpyToHost(hostOut, devOut, outBytes);
            float[] outRes = hostOut.toArray(ValueLayout.JAVA_FLOAT);
            float[] expectedOut = {
                1.1f, 1.2f, 1.3f,
                3.1f, 3.2f, 3.3f
            };
            assertArrayEquals(expectedOut, outRes, 1e-6f);

            // Backward
            float[] outGrad = {
                0.5f, 0.5f, 0.5f,
                1.0f, 1.0f, 1.0f
            };
            MemorySegment hostOutGrad = arena.allocateFrom(ValueLayout.JAVA_FLOAT, outGrad);
            MemorySegment devOutGrad = CudaBridge.cudaMalloc(outBytes);
            CudaBridge.cudaMemcpyToDevice(devOutGrad, hostOutGrad, outBytes);

            MemorySegment devEmbGrad = CudaBridge.cudaMalloc(embBytes);
            // Zero-initialize devEmbGrad
            float[] zeros = new float[vocabSize * embDim];
            MemorySegment hostZeros = arena.allocateFrom(ValueLayout.JAVA_FLOAT, zeros);
            CudaBridge.cudaMemcpyToDevice(devEmbGrad, hostZeros, embBytes);

            CudaBridge.cudaEmbeddingBackward(devOutGrad, devIds, devEmbGrad, numTokens, embDim);

            MemorySegment hostEmbGrad = arena.allocate(ValueLayout.JAVA_FLOAT, vocabSize * embDim);
            CudaBridge.cudaMemcpyToHost(hostEmbGrad, devEmbGrad, embBytes);
            float[] embGradRes = hostEmbGrad.toArray(ValueLayout.JAVA_FLOAT);

            // Expect gradient at index 1 to be outGrad[0..2], index 3 to be outGrad[3..5], others 0
            float[] expectedEmbGrad = {
                0.0f, 0.0f, 0.0f,
                0.5f, 0.5f, 0.5f,
                0.0f, 0.0f, 0.0f,
                1.0f, 1.0f, 1.0f
            };
            assertArrayEquals(expectedEmbGrad, embGradRes, 1e-6f);

            CudaBridge.cudaFree(devOutGrad);
            CudaBridge.cudaFree(devEmbGrad);
        } finally {
            CudaBridge.cudaFree(devEmb);
            CudaBridge.cudaFree(devIds);
            CudaBridge.cudaFree(devOut);
        }
    }

    @Test
    public void testCudaAttention() {
        int rows = 2;
        int cols = 2;
        long byteSize = rows * cols * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devScores = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devDA = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devDS = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            // Causal masking & softmax test:
            // scores:
            // [ 1.0, 2.0 ] -> masked to [ 1.0, -1e9 ] -> scaled (inv_scale = 1.0) -> softmax is [1.0, 0.0] (clamped to 1.0-1e-15, 1e-15)
            // [ 3.0, 3.0 ] -> masked to [ 3.0, 3.0 ] -> scaled (inv_scale = 1.0) -> softmax is [0.5, 0.5]
            float[] scores = {
                1.0f, 2.0f,
                3.0f, 3.0f
            };

            MemorySegment hostScores = arena.allocateFrom(ValueLayout.JAVA_FLOAT, scores);
            MemorySegment hostA = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);

            CudaBridge.cudaMemcpyToDevice(devScores, hostScores, byteSize);

            // Forward
            CudaBridge.cudaAttentionForward(devScores, rows, cols, 1.0f);

            CudaBridge.cudaMemcpyToHost(hostA, devScores, byteSize);
            float[] aRes = hostA.toArray(ValueLayout.JAVA_FLOAT);

            // Check Row 0: first element should be close to 1.0, second element close to 1e-15
            assertEquals(1.0f - 1e-15f, aRes[0], 1e-5f);
            assertEquals(1e-15f, aRes[1], 1e-5f);
            // Check Row 1: elements should be 0.5, 0.5
            assertEquals(0.5f, aRes[2], 1e-5f);
            assertEquals(0.5f, aRes[3], 1e-5f);

            // Backward:
            // A = [1.0, 0.0; 0.5, 0.5]
            // dA = [1.0, 1.0; 2.0, 4.0]
            // Scale = 2.0
            // Row 0 dot product = dA[0]*A[0] = 1.0 * 1.0 = 1.0. For col 0: A[0]*(dA[0]-dot)*scale = 1.0*(1.0-1.0)*2.0 = 0.0
            // Row 1 dot product = dA[2]*A[2] + dA[3]*A[3] = 2.0*0.5 + 4.0*0.5 = 3.0.
            // For col 0: A[2]*(dA[2]-dot)*scale = 0.5*(2.0-3.0)*2.0 = -1.0.
            // For col 1: A[3]*(dA[3]-dot)*scale = 0.5*(4.0-3.0)*2.0 = 1.0.
            float[] dAData = {
                1.0f, 1.0f,
                2.0f, 4.0f
            };
            MemorySegment hostDA = arena.allocateFrom(ValueLayout.JAVA_FLOAT, dAData);
            MemorySegment hostDS = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);

            CudaBridge.cudaMemcpyToDevice(devA, hostA, byteSize);
            CudaBridge.cudaMemcpyToDevice(devDA, hostDA, byteSize);

            CudaBridge.cudaAttentionBackward(devA, devDA, devDS, rows, cols, 2.0f);

            CudaBridge.cudaMemcpyToHost(hostDS, devDS, byteSize);
            float[] dsRes = hostDS.toArray(ValueLayout.JAVA_FLOAT);

            assertEquals(0.0f, dsRes[0], 1e-5f);
            assertEquals(0.0f, dsRes[1], 1e-5f); // Masked element is 0.0
            assertEquals(-1.0f, dsRes[2], 1e-5f);
            assertEquals(1.0f, dsRes[3], 1e-5f);

        } finally {
            CudaBridge.cudaFree(devScores);
            CudaBridge.cudaFree(devA);
            CudaBridge.cudaFree(devDA);
            CudaBridge.cudaFree(devDS);
        }
    }

    @Test
    public void testCudaSoftmax() {
        int rows = 1;
        int cols = 3;
        long byteSize = rows * cols * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devInput = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devOutput = CudaBridge.cudaMalloc(byteSize);

        try (Arena arena = Arena.ofConfined()) {
            float[] input = {1.0f, 2.0f, 3.0f};
            // Softmax should be exp(1)/sum, exp(2)/sum, exp(3)/sum
            float sum = (float) (Math.exp(1) + Math.exp(2) + Math.exp(3));
            float[] expected = {
                (float) Math.exp(1) / sum,
                (float) Math.exp(2) / sum,
                (float) Math.exp(3) / sum
            };

            MemorySegment hostInput = arena.allocateFrom(ValueLayout.JAVA_FLOAT, input);
            MemorySegment hostOutput = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);

            CudaBridge.cudaMemcpyToDevice(devInput, hostInput, byteSize);

            CudaBridge.cudaSoftmaxForward(devInput, devOutput, rows, cols);

            CudaBridge.cudaMemcpyToHost(hostOutput, devOutput, byteSize);
            float[] result = hostOutput.toArray(ValueLayout.JAVA_FLOAT);

            assertArrayEquals(expected, result, 1e-5f);
        } finally {
            CudaBridge.cudaFree(devInput);
            CudaBridge.cudaFree(devOutput);
        }
    }

    @Test
    public void testCudaLayerNorm() {
        int rows = 2;
        int cols = 2;
        long byteSize = rows * cols * ValueLayout.JAVA_FLOAT.byteSize();
        long weightBytes = cols * ValueLayout.JAVA_FLOAT.byteSize();
        long statBytes = rows * ValueLayout.JAVA_FLOAT.byteSize();

        MemorySegment devInput = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devGamma = CudaBridge.cudaMalloc(weightBytes);
        MemorySegment devBeta = CudaBridge.cudaMalloc(weightBytes);
        MemorySegment devOutput = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devXHat = CudaBridge.cudaMalloc(byteSize);
        MemorySegment devMean = CudaBridge.cudaMalloc(statBytes);
        MemorySegment devVar = CudaBridge.cudaMalloc(statBytes);

        try (Arena arena = Arena.ofConfined()) {
            float[] input = {
                1.0f, 2.0f,
                10.0f, 20.0f
            };
            float[] gamma = {1.0f, 2.0f};
            float[] beta = {0.5f, -0.5f};
            float[] mean = {1.5f, 15.0f};
            float[] var = {0.25f, 25.0f}; // standard dev is 0.5 and 5.0 respectively

            MemorySegment hostInput = arena.allocateFrom(ValueLayout.JAVA_FLOAT, input);
            MemorySegment hostGamma = arena.allocateFrom(ValueLayout.JAVA_FLOAT, gamma);
            MemorySegment hostBeta = arena.allocateFrom(ValueLayout.JAVA_FLOAT, beta);
            MemorySegment hostMean = arena.allocateFrom(ValueLayout.JAVA_FLOAT, mean);
            MemorySegment hostVar = arena.allocateFrom(ValueLayout.JAVA_FLOAT, var);

            MemorySegment hostOutput = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);
            MemorySegment hostXHat = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);

            CudaBridge.cudaMemcpyToDevice(devInput, hostInput, byteSize);
            CudaBridge.cudaMemcpyToDevice(devGamma, hostGamma, weightBytes);
            CudaBridge.cudaMemcpyToDevice(devBeta, hostBeta, weightBytes);
            CudaBridge.cudaMemcpyToDevice(devMean, hostMean, statBytes);
            CudaBridge.cudaMemcpyToDevice(devVar, hostVar, statBytes);

            // Forward
            float eps = 1e-9f;
            CudaBridge.cudaLayerNormForward(devInput, devGamma, devBeta, devOutput, devXHat, devMean, devVar, rows, cols, eps);

            CudaBridge.cudaMemcpyToHost(hostOutput, devOutput, byteSize);
            CudaBridge.cudaMemcpyToHost(hostXHat, devXHat, byteSize);

            float[] outRes = hostOutput.toArray(ValueLayout.JAVA_FLOAT);
            float[] xHatRes = hostXHat.toArray(ValueLayout.JAVA_FLOAT);

            // Row 0:
            // xHat = [(1 - 1.5)/0.5, (2 - 1.5)/0.5] = [-1.0, 1.0]
            // output = [xHat[0]*1.0 + 0.5, xHat[1]*2.0 - 0.5] = [-0.5, 1.5]
            // Row 1:
            // xHat = [(10 - 15)/5, (20 - 15)/5] = [-1.0, 1.0]
            // output = [-0.5, 1.5]
            float[] expectedXHat = {-1.0f, 1.0f, -1.0f, 1.0f};
            float[] expectedOutput = {-0.5f, 1.5f, -0.5f, 1.5f};

            assertArrayEquals(expectedXHat, xHatRes, 1e-5f);
            assertArrayEquals(expectedOutput, outRes, 1e-5f);

            // Backward
            float[] outGrad = {
                1.0f, 1.0f,
                1.0f, 1.0f
            };
            MemorySegment hostOutGrad = arena.allocateFrom(ValueLayout.JAVA_FLOAT, outGrad);
            MemorySegment devOutGrad = CudaBridge.cudaMalloc(byteSize);
            MemorySegment devDInput = CudaBridge.cudaMalloc(byteSize);
            MemorySegment devDGamma = CudaBridge.cudaMalloc(weightBytes);
            MemorySegment devDBeta = CudaBridge.cudaMalloc(weightBytes);

            CudaBridge.cudaMemcpyToDevice(devOutGrad, hostOutGrad, byteSize);

            // Zero-initialize devDGamma and devDBeta
            float[] zeros = {0.0f, 0.0f};
            MemorySegment hostZeros = arena.allocateFrom(ValueLayout.JAVA_FLOAT, zeros);
            CudaBridge.cudaMemcpyToDevice(devDGamma, hostZeros, weightBytes);
            CudaBridge.cudaMemcpyToDevice(devDBeta, hostZeros, weightBytes);

            CudaBridge.cudaLayerNormBackward(devOutGrad, devXHat, devVar, devGamma, devDInput, devDGamma, devDBeta, rows, cols, eps);

            MemorySegment hostDInput = arena.allocate(ValueLayout.JAVA_FLOAT, rows * cols);
            MemorySegment hostDGamma = arena.allocate(ValueLayout.JAVA_FLOAT, cols);
            MemorySegment hostDBeta = arena.allocate(ValueLayout.JAVA_FLOAT, cols);

            CudaBridge.cudaMemcpyToHost(hostDInput, devDInput, byteSize);
            CudaBridge.cudaMemcpyToHost(hostDGamma, devDGamma, weightBytes);
            CudaBridge.cudaMemcpyToHost(hostDBeta, devDBeta, weightBytes);

            float[] dInputRes = hostDInput.toArray(ValueLayout.JAVA_FLOAT);
            float[] dGammaRes = hostDGamma.toArray(ValueLayout.JAVA_FLOAT);
            float[] dBetaRes = hostDBeta.toArray(ValueLayout.JAVA_FLOAT);

            // dInput should be close to 0.0
            float[] expectedDInput = {0.0f, 0.0f, 0.0f, 0.0f};
            assertArrayEquals(expectedDInput, dInputRes, 1e-5f);

            // dGamma should be [-2.0, 2.0]
            float[] expectedDGamma = {-2.0f, 2.0f};
            assertArrayEquals(expectedDGamma, dGammaRes, 1e-5f);

            // dBeta should be [2.0, 2.0]
            float[] expectedDBeta = {2.0f, 2.0f};
            assertArrayEquals(expectedDBeta, dBetaRes, 1e-5f);

            CudaBridge.cudaFree(devOutGrad);
            CudaBridge.cudaFree(devDInput);
            CudaBridge.cudaFree(devDGamma);
            CudaBridge.cudaFree(devDBeta);
        } finally {
            CudaBridge.cudaFree(devInput);
            CudaBridge.cudaFree(devGamma);
            CudaBridge.cudaFree(devBeta);
            CudaBridge.cudaFree(devOutput);
            CudaBridge.cudaFree(devXHat);
            CudaBridge.cudaFree(devMean);
            CudaBridge.cudaFree(devVar);
        }
    }
}
