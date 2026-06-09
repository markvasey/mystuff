package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AdamTest {

    @Test
    public void testAdamUpdate() {
        // Initialize weights with 1.0, gradient with 0.5
        Matrix weights = new Matrix(1, 1);
        weights.set(0, 0, 1.0f);
        Matrix gradient = new Matrix(1, 1);
        gradient.set(0, 0, 0.5f);

        Adam opt = new Adam(1, 1);
        
        // Run first update step with lr = 0.1f
        // For step 1:
        // m_1 = 0.9 * 0 + (1-0.9) * 0.5 = 0.05
        // v_1 = 0.999 * 0 + (1-0.999) * 0.25 = 0.00025
        // m_hat = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
        // v_hat = 0.00025 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
        // delta_w = -0.1 * 0.5 / (sqrt(0.25) + 1e-8) = -0.1 * 0.5 / 0.5 = -0.1
        // w_new = 1.0 - 0.1 = 0.9
        opt.update(weights, gradient, 0.1f);
        assertEquals(0.9f, weights.get(0, 0), 1e-6f);
    }

    @Test
    public void testGpuAdamMatchesCpu() {
        Matrix cpuWeights = new Matrix(2, 3);
        float[] wData = {1.0f, 2.0f, -0.5f, 0.0f, 3.5f, -2.0f};
        System.arraycopy(wData, 0, cpuWeights.getData(), 0, wData.length);
        
        Matrix cpuGrad = new Matrix(2, 3);
        float[] gData = {0.1f, -0.2f, 0.05f, 0.9f, -0.01f, 0.4f};
        System.arraycopy(gData, 0, cpuGrad.getData(), 0, gData.length);

        Adam cpuOpt = new Adam(2, 3);
        GpuAdam gpuOpt = new GpuAdam(2, 3);
        
        try (GpuMatrix gpuWeights = GpuMatrix.fromCpu(cpuWeights);
             GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad)) {
             
             cpuOpt.update(cpuWeights, cpuGrad, 0.01f);
             gpuOpt.update(gpuWeights, gpuGrad, 0.01f);
             
             Matrix back = gpuWeights.toCpu();
             assertArrayEquals(cpuWeights.getData(), back.getData(), 1e-5f);
             
             gpuOpt.close();
        }
    }
}
