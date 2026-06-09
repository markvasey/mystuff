package com.learnai.words.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class GpuMatrixTest {

    @Test
    public void testUploadAndDownload() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        try (GpuMatrix m = new GpuMatrix(2, 3)) {
            m.upload(data);

            float[] result = new float[6];
            m.download(result);

            assertArrayEquals(data, result, 1e-6f);
        }
    }

    @Test
    public void testMatrixMultiply() {
        // A: 2x3, B: 3x2 -> C: 2x2
        float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        float[] dataB = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

        try (GpuMatrix a = new GpuMatrix(2, 3);
             GpuMatrix b = new GpuMatrix(3, 2)) {
            a.upload(dataA);
            b.upload(dataB);

            try (GpuMatrix c = a.multiply(b)) {
                assertEquals(2, c.getRows());
                assertEquals(2, c.getCols());

                float[] result = new float[4];
                c.download(result);

                float[] expected = {58.0f, 64.0f, 139.0f, 154.0f};
                assertArrayEquals(expected, result, 1e-5f);
            }
        }
    }

    @Test
    public void testAddAndSubtract() {
        float[] dataA = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] dataB = {0.5f, 1.5f, 2.5f, 3.5f};

        try (GpuMatrix a = new GpuMatrix(2, 2);
             GpuMatrix b = new GpuMatrix(2, 2)) {
            a.upload(dataA);
            b.upload(dataB);

            // Add
            try (GpuMatrix c = a.add(b)) {
                float[] res = new float[4];
                c.download(res);
                assertArrayEquals(new float[]{1.5f, 3.5f, 5.5f, 7.5f}, res, 1e-6f);
            }

            // Subtract
            try (GpuMatrix d = a.subtract(b)) {
                float[] res = new float[4];
                d.download(res);
                assertArrayEquals(new float[]{0.5f, 0.5f, 0.5f, 0.5f}, res, 1e-6f);
            }
        }
    }

    @Test
    public void testElementWiseMultiplySquareAndSqrt() {
        float[] data = {4.0f, 9.0f, 16.0f};
        try (GpuMatrix a = new GpuMatrix(1, 3);
             GpuMatrix b = new GpuMatrix(1, 3)) {
            a.upload(data);
            b.upload(new float[]{2.0f, 3.0f, 4.0f});

            // Multiply element-wise
            try (GpuMatrix c = a.multiplyElementWise(b)) {
                float[] res = new float[3];
                c.download(res);
                assertArrayEquals(new float[]{8.0f, 27.0f, 64.0f}, res, 1e-5f);
            }

            // Square
            try (GpuMatrix d = b.square()) {
                float[] res = new float[3];
                d.download(res);
                assertArrayEquals(new float[]{4.0f, 9.0f, 16.0f}, res, 1e-6f);
            }

            // Sqrt
            try (GpuMatrix e = a.sqrt(0.0f)) {
                float[] res = new float[3];
                e.download(res);
                assertArrayEquals(new float[]{2.0f, 3.0f, 4.0f}, res, 1e-6f);
            }
        }
    }

    @Test
    public void testRowMeanAndVariance() {
        float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f};
        try (GpuMatrix a = new GpuMatrix(2, 3)) {
            a.upload(data);

            try (GpuMatrix mean = a.rowMean();
                 GpuMatrix var = a.rowVariance(mean)) {
                float[] mRes = new float[2];
                float[] vRes = new float[2];
                mean.download(mRes);
                var.download(vRes);

                assertArrayEquals(new float[]{2.0f, 6.0f}, mRes, 1e-6f);
                assertArrayEquals(new float[]{2.0f / 3.0f, 8.0f / 3.0f}, vRes, 1e-5f);
            }
        }
    }

    @Test
    public void testTranspose() {
        float[] data = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };
        try (GpuMatrix src = new GpuMatrix(2, 3)) {
            src.upload(data);
            try (GpuMatrix dest = src.transpose()) {
                assertEquals(3, dest.getRows());
                assertEquals(2, dest.getCols());

                float[] result = new float[6];
                dest.download(result);

                float[] expected = {
                    1.0f, 4.0f,
                    2.0f, 5.0f,
                    3.0f, 6.0f
                };
                assertArrayEquals(expected, result, 1e-6f);
            }
        }
    }

    @Test
    public void testCpuBridgeHelpers() {
        Matrix cpuMat = new Matrix(2, 2);
        cpuMat.set(0, 0, 10.0f);
        cpuMat.set(0, 1, 20.0f);
        cpuMat.set(1, 0, 30.0f);
        cpuMat.set(1, 1, 40.0f);

        try (GpuMatrix gpuMat = GpuMatrix.fromCpu(cpuMat)) {
            assertEquals(2, gpuMat.getRows());
            assertEquals(2, gpuMat.getCols());

            Matrix back = gpuMat.toCpu();
            assertEquals(10.0f, back.get(0, 0));
            assertEquals(20.0f, back.get(0, 1));
            assertEquals(30.0f, back.get(1, 0));
            assertEquals(40.0f, back.get(1, 1));
        }
    }

    @Test
    public void testGCDeallocationAndCleaner() {
        // Create matrices and let them drop out of scope to test cleaner safety
        for (int i = 0; i < 50; i++) {
            GpuMatrix m = new GpuMatrix(10, 10);
            float[] d = new float[100];
            m.upload(d);
            // Let the object drop out of scope without calling close
        }
        // Force garbage collection to trigger cleaners
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            // Ignored
        }
    }
}
