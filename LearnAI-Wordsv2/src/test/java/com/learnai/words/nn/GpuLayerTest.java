package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.Matrix;
import org.junit.jupiter.api.Test;
import java.io.*;
import static org.junit.jupiter.api.Assertions.*;

public class GpuLayerTest {

    @Test
    public void testGpuDenseLayerMatchesCpu() throws Exception {
        int inDim = 3;
        int outDim = 2;

        DenseLayer cpuLayer = new DenseLayer(inDim, outDim);
        GpuDenseLayer gpuLayer = new GpuDenseLayer(inDim, outDim);

        // Sync weights and biases by serialization/deserialization
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(baos);
        cpuLayer.save(dos);
        dos.flush();

        ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
        DataInputStream dis = new DataInputStream(bais);
        gpuLayer.load(dis);

        // Prepare input
        Matrix cpuInput = new Matrix(4, inDim);
        float[] inputData = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            -1.0f, 0.0f, 1.0f,
            0.5f, -0.5f, 0.2f
        };
        System.arraycopy(inputData, 0, cpuInput.getData(), 0, inputData.length);
        GpuMatrix gpuInput = GpuMatrix.fromCpu(cpuInput);

        // Forward Pass
        Layer.ForwardResult cpuRes = cpuLayer.forward(cpuInput);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(gpuInput);

        Matrix gpuOutputCpu = gpuRes.output.toCpu();
        assertArrayEquals(cpuRes.output.getData(), gpuOutputCpu.getData(), 1e-5f);

        // Backward Pass
        Matrix cpuGrad = new Matrix(4, outDim);
        float[] gradData = {
            0.1f, -0.2f,
            0.5f, 0.4f,
            -0.3f, 0.2f,
            0.0f, -0.1f
        };
        System.arraycopy(gradData, 0, cpuGrad.getData(), 0, gradData.length);
        GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad);

        float lr = 0.01f;
        Matrix cpuInGrad = cpuLayer.backward(cpuGrad, cpuRes.context, lr);
        GpuMatrix gpuInGrad = gpuLayer.backward(gpuGrad, gpuRes.context, lr);

        Matrix gpuInGradCpu = gpuInGrad.toCpu();
        assertArrayEquals(cpuInGrad.getData(), gpuInGradCpu.getData(), 1e-5f);

        // Verify updated weights and biases match numerically
        ByteArrayOutputStream baos2 = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos2));
        
        ByteArrayOutputStream baosGpu = new ByteArrayOutputStream();
        gpuLayer.save(new DataOutputStream(baosGpu));

        DataInputStream disCpu = new DataInputStream(new ByteArrayInputStream(baos2.toByteArray()));
        DataInputStream disGpu = new DataInputStream(new ByteArrayInputStream(baosGpu.toByteArray()));

        Matrix cpuW = Matrix.load(disCpu);
        Matrix cpuB = Matrix.load(disCpu);
        Matrix gpuW = Matrix.load(disGpu);
        Matrix gpuB = Matrix.load(disGpu);

        assertArrayEquals(cpuW.getData(), gpuW.getData(), 1e-5f);
        assertArrayEquals(cpuB.getData(), gpuB.getData(), 1e-5f);

        // Clean up
        gpuInput.close();
        gpuRes.output.close();
        gpuGrad.close();
        gpuInGrad.close();
        gpuLayer.close();
    }

    @Test
    public void testGpuLayerNormMatchesCpu() throws Exception {
        int dim = 4;
        LayerNorm cpuLayer = new LayerNorm(dim);
        GpuLayerNorm gpuLayer = new GpuLayerNorm(dim);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos));
        gpuLayer.load(new DataInputStream(new ByteArrayInputStream(baos.toByteArray())));

        Matrix cpuInput = new Matrix(3, dim);
        float[] inputData = {
            1.0f, 2.0f, 3.0f, 4.0f,
            0.5f, 0.5f, 0.5f, 0.5f,
            -10.0f, 0.0f, 10.0f, 5.0f
        };
        System.arraycopy(inputData, 0, cpuInput.getData(), 0, inputData.length);
        GpuMatrix gpuInput = GpuMatrix.fromCpu(cpuInput);

        // Forward
        Layer.ForwardResult cpuRes = cpuLayer.forward(cpuInput);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(gpuInput);

        Matrix gpuOutCpu = gpuRes.output.toCpu();
        assertArrayEquals(cpuRes.output.getData(), gpuOutCpu.getData(), 1e-5f);

        // Backward
        Matrix cpuGrad = new Matrix(3, dim);
        float[] gradData = {
            0.1f, -0.2f, 0.3f, -0.4f,
            0.5f, 0.4f, 0.3f, 0.2f,
            -0.1f, 0.0f, 0.1f, 0.2f
        };
        System.arraycopy(gradData, 0, cpuGrad.getData(), 0, gradData.length);
        GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad);

        float lr = 0.01f;
        Matrix cpuInGrad = cpuLayer.backward(cpuGrad, cpuRes.context, lr);
        GpuMatrix gpuInGrad = gpuLayer.backward(gpuGrad, gpuRes.context, lr);

        Matrix gpuInGradCpu = gpuInGrad.toCpu();
        assertArrayEquals(cpuInGrad.getData(), gpuInGradCpu.getData(), 1e-5f);

        // Verify updated gamma and beta match numerically
        ByteArrayOutputStream baos2 = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos2));
        
        ByteArrayOutputStream baosGpu = new ByteArrayOutputStream();
        gpuLayer.save(new DataOutputStream(baosGpu));

        DataInputStream disCpu = new DataInputStream(new ByteArrayInputStream(baos2.toByteArray()));
        DataInputStream disGpu = new DataInputStream(new ByteArrayInputStream(baosGpu.toByteArray()));

        Matrix cpuG = Matrix.load(disCpu);
        Matrix cpuB = Matrix.load(disCpu);
        Matrix gpuG = Matrix.load(disGpu);
        Matrix gpuB = Matrix.load(disGpu);

        assertArrayEquals(cpuG.getData(), gpuG.getData(), 1e-5f);
        assertArrayEquals(cpuB.getData(), gpuB.getData(), 1e-5f);

        // Clean up
        gpuInput.close();
        gpuRes.output.close();
        gpuGrad.close();
        gpuInGrad.close();
        gpuLayer.close();
    }

    @Test
    public void testGpuSoftmaxMatchesCpu() {
        GpuSoftmaxLayer gpuLayer = new GpuSoftmaxLayer();
        SoftmaxLayer cpuLayer = new SoftmaxLayer();

        Matrix cpuInput = new Matrix(2, 3);
        float[] data = {
            1.0f, 2.0f, 3.0f,
            -1.0f, -2.0f, 5.0f
        };
        System.arraycopy(data, 0, cpuInput.getData(), 0, data.length);
        GpuMatrix gpuInput = GpuMatrix.fromCpu(cpuInput);

        // Forward
        Layer.ForwardResult cpuRes = cpuLayer.forward(cpuInput);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(gpuInput);

        assertArrayEquals(cpuRes.output.getData(), gpuRes.output.toCpu().getData(), 1e-5f);

        // Backward
        Matrix cpuTarget = new Matrix(2, 3);
        float[] targetData = {
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f
        };
        System.arraycopy(targetData, 0, cpuTarget.getData(), 0, targetData.length);
        GpuMatrix gpuTarget = GpuMatrix.fromCpu(cpuTarget);

        Matrix cpuInGrad = cpuLayer.backward(cpuTarget, cpuRes.context, 0.0f);
        GpuMatrix gpuInGrad = gpuLayer.backward(gpuTarget, gpuRes.context, 0.0f);

        assertArrayEquals(cpuInGrad.getData(), gpuInGrad.toCpu().getData(), 1e-5f);

        // Clean up
        gpuInput.close();
        gpuRes.output.close();
        gpuTarget.close();
        gpuInGrad.close();
    }

    @Test
    public void testGpuEmbeddingMatchesCpu() throws Exception {
        int vocabSize = 10;
        int dim = 4;
        EmbeddingLayer cpuLayer = new EmbeddingLayer(vocabSize, dim);
        GpuEmbeddingLayer gpuLayer = new GpuEmbeddingLayer(vocabSize, dim);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos));
        gpuLayer.load(new DataInputStream(new ByteArrayInputStream(baos.toByteArray())));

        int[] tokenIds = {1, 3, 0, 9, 2};

        // Forward
        Layer.ForwardResult cpuRes = cpuLayer.forward(tokenIds);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(tokenIds);

        assertArrayEquals(cpuRes.output.getData(), gpuRes.output.toCpu().getData(), 1e-5f);

        // Backward
        Matrix cpuGrad = new Matrix(tokenIds.length, dim);
        float[] gradData = new float[tokenIds.length * dim];
        for (int i = 0; i < gradData.length; i++) gradData[i] = i * 0.1f;
        System.arraycopy(gradData, 0, cpuGrad.getData(), 0, gradData.length);
        GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad);

        float lr = 0.05f;
        cpuLayer.backward(cpuGrad, cpuRes.context, lr);
        gpuLayer.backward(gpuGrad, gpuRes.context, lr);

        // Verify weights match after updates
        ByteArrayOutputStream baos2 = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos2));
        
        ByteArrayOutputStream baosGpu = new ByteArrayOutputStream();
        gpuLayer.save(new DataOutputStream(baosGpu));

        DataInputStream disCpu = new DataInputStream(new ByteArrayInputStream(baos2.toByteArray()));
        DataInputStream disGpu = new DataInputStream(new ByteArrayInputStream(baosGpu.toByteArray()));

        Matrix cpuEmb = Matrix.load(disCpu);
        Matrix gpuEmb = Matrix.load(disGpu);

        assertArrayEquals(cpuEmb.getData(), gpuEmb.getData(), 1e-5f);

        gpuRes.output.close();
        gpuGrad.close();
        gpuLayer.close();
    }

    @Test
    public void testGpuCausalAttentionMatchesCpu() throws Exception {
        int dim = 3;
        CausalSelfAttentionLayer cpuLayer = new CausalSelfAttentionLayer(dim);
        GpuCausalSelfAttentionLayer gpuLayer = new GpuCausalSelfAttentionLayer(dim);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos));
        gpuLayer.load(new DataInputStream(new ByteArrayInputStream(baos.toByteArray())));

        Matrix cpuInput = new Matrix(2, dim);
        float[] data = {
            1.0f, -0.5f, 2.0f,
            0.5f, 1.5f, -1.0f
        };
        System.arraycopy(data, 0, cpuInput.getData(), 0, data.length);
        GpuMatrix gpuInput = GpuMatrix.fromCpu(cpuInput);

        // Forward
        Layer.ForwardResult cpuRes = cpuLayer.forward(cpuInput);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(gpuInput);

        assertArrayEquals(cpuRes.output.getData(), gpuRes.output.toCpu().getData(), 1e-5f);

        // Backward
        Matrix cpuGrad = new Matrix(2, dim);
        float[] gradData = {
            0.1f, 0.2f, -0.3f,
            -0.4f, 0.5f, 0.1f
        };
        System.arraycopy(gradData, 0, cpuGrad.getData(), 0, gradData.length);
        GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad);

        float lr = 0.02f;
        Matrix cpuInGrad = cpuLayer.backward(cpuGrad, cpuRes.context, lr);
        GpuMatrix gpuInGrad = gpuLayer.backward(gpuGrad, gpuRes.context, lr);

        assertArrayEquals(cpuInGrad.getData(), gpuInGrad.toCpu().getData(), 1e-5f);

        // Verify updated weights match
        ByteArrayOutputStream baos2 = new ByteArrayOutputStream();
        cpuLayer.save(new DataOutputStream(baos2));
        
        ByteArrayOutputStream baosGpu = new ByteArrayOutputStream();
        gpuLayer.save(new DataOutputStream(baosGpu));

        DataInputStream disCpu = new DataInputStream(new ByteArrayInputStream(baos2.toByteArray()));
        DataInputStream disGpu = new DataInputStream(new ByteArrayInputStream(baosGpu.toByteArray()));

        Matrix cpuWq = Matrix.load(disCpu);
        Matrix cpuWk = Matrix.load(disCpu);
        Matrix cpuWv = Matrix.load(disCpu);
        
        Matrix gpuWq = Matrix.load(disGpu);
        Matrix gpuWk = Matrix.load(disGpu);
        Matrix gpuWv = Matrix.load(disGpu);

        assertArrayEquals(cpuWq.getData(), gpuWq.getData(), 1e-5f);
        assertArrayEquals(cpuWk.getData(), gpuWk.getData(), 1e-5f);
        assertArrayEquals(cpuWv.getData(), gpuWv.getData(), 1e-5f);

        gpuInput.close();
        gpuRes.output.close();
        gpuGrad.close();
        gpuInGrad.close();
        gpuLayer.close();
    }

    @Test
    public void testGpuPositionalEncodingMatchesCpu() throws Exception {
        int seqLen = 3;
        int dModel = 4;
        PositionalEncoding cpuLayer = new PositionalEncoding(seqLen, dModel);
        GpuPositionalEncoding gpuLayer = new GpuPositionalEncoding(seqLen, dModel);

        Matrix cpuInput = new Matrix(seqLen, dModel);
        float[] inputData = {
            0.5f, -0.5f, 1.0f, 2.0f,
            1.5f, 0.0f, -1.0f, -2.0f,
            0.1f, 0.2f, 0.3f, 0.4f
        };
        System.arraycopy(inputData, 0, cpuInput.getData(), 0, inputData.length);
        GpuMatrix gpuInput = GpuMatrix.fromCpu(cpuInput);

        // Forward
        Layer.ForwardResult cpuRes = cpuLayer.forward(cpuInput);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(gpuInput);

        Matrix gpuOutCpu = gpuRes.output.toCpu();
        assertArrayEquals(cpuRes.output.getData(), gpuOutCpu.getData(), 1e-5f);

        // Backward
        Matrix cpuGrad = new Matrix(seqLen, dModel);
        float[] gradData = {
            0.1f, 0.2f, 0.3f, 0.4f,
            -0.1f, -0.2f, -0.3f, -0.4f,
            0.5f, 0.5f, 0.5f, 0.5f
        };
        System.arraycopy(gradData, 0, cpuGrad.getData(), 0, gradData.length);
        GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad);

        float lr = 0.01f;
        Matrix cpuInGrad = cpuLayer.backward(cpuGrad, cpuRes.context, lr);
        GpuMatrix gpuInGrad = gpuLayer.backward(gpuGrad, gpuRes.context, lr);

        Matrix gpuInGradCpu = gpuInGrad.toCpu();
        assertArrayEquals(cpuInGrad.getData(), gpuInGradCpu.getData(), 1e-5f);

        gpuInput.close();
        gpuRes.output.close();
        gpuGrad.close();
        gpuInGrad.close();
        gpuLayer.close();
    }

    @Test
    public void testGpuResidualBlockMatchesCpu() throws Exception {
        int dim = 3;
        DenseLayer cpuSub = new DenseLayer(dim, dim);
        GpuDenseLayer gpuSub = new GpuDenseLayer(dim, dim);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        cpuSub.save(new DataOutputStream(baos));
        gpuSub.load(new DataInputStream(new ByteArrayInputStream(baos.toByteArray())));

        ResidualBlock cpuLayer = new ResidualBlock(cpuSub);
        GpuResidualBlock gpuLayer = new GpuResidualBlock(gpuSub);

        Matrix cpuInput = new Matrix(2, dim);
        float[] inputData = {
            1.0f, -0.5f, 2.0f,
            0.5f, 1.5f, -1.0f
        };
        System.arraycopy(inputData, 0, cpuInput.getData(), 0, inputData.length);
        GpuMatrix gpuInput = GpuMatrix.fromCpu(cpuInput);

        // Forward
        Layer.ForwardResult cpuRes = cpuLayer.forward(cpuInput);
        GpuLayer.GpuForwardResult gpuRes = gpuLayer.forward(gpuInput);

        Matrix gpuOutCpu = gpuRes.output.toCpu();
        assertArrayEquals(cpuRes.output.getData(), gpuOutCpu.getData(), 1e-5f);

        // Backward
        Matrix cpuGrad = new Matrix(2, dim);
        float[] gradData = {
            0.1f, 0.2f, -0.3f,
            -0.4f, 0.5f, 0.1f
        };
        System.arraycopy(gradData, 0, cpuGrad.getData(), 0, gradData.length);
        GpuMatrix gpuGrad = GpuMatrix.fromCpu(cpuGrad);

        float lr = 0.02f;
        Matrix cpuInGrad = cpuLayer.backward(cpuGrad, cpuRes.context, lr);
        GpuMatrix gpuInGrad = gpuLayer.backward(gpuGrad, gpuRes.context, lr);

        Matrix gpuInGradCpu = gpuInGrad.toCpu();
        assertArrayEquals(cpuInGrad.getData(), gpuInGradCpu.getData(), 1e-5f);

        gpuInput.close();
        gpuRes.output.close();
        gpuGrad.close();
        gpuInGrad.close();
        gpuLayer.close();
    }
}
