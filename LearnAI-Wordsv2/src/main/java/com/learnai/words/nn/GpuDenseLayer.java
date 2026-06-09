package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GpuDenseLayer implements GpuLayer {
    private GpuMatrix weights;
    private GpuMatrix bias;
    private final GpuAdam weightsOpt;
    private final GpuAdam biasOpt;

    public GpuDenseLayer(int inputDim, int outputDim) {
        Matrix cpuW = Matrix.random(inputDim, outputDim);
        this.weights = GpuMatrix.fromCpu(cpuW);

        this.bias = new GpuMatrix(1, outputDim);
        float[] zeroBias = new float[outputDim];
        this.bias.upload(zeroBias);

        this.weightsOpt = new GpuAdam(inputDim, outputDim);
        this.biasOpt = new GpuAdam(1, outputDim);
    }

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        // Output = input * weights + bias
        GpuMatrix temp = input.multiply(weights);
        temp.addInPlace(bias);
        return new GpuForwardResult(temp, input);
    }

    @Override
    public GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate) {
        GpuMatrix lastInput = (GpuMatrix) context;

        // weightsGradient = lastInput^T * outputGradient
        GpuMatrix weightsGradient = lastInput.multiply(outputGradient, true, false);

        // biasGradient = sum of outputGradient along the row dimension.
        // We compute this using U^T * outputGradient, where U is a vector of ones of size (ogRows, 1).
        int ogRows = outputGradient.getRows();
        int ogCols = outputGradient.getCols();
        GpuMatrix biasGradient;
        try (GpuMatrix ones = new GpuMatrix(ogRows, 1)) {
            float[] onesData = new float[ogRows];
            for (int i = 0; i < ogRows; i++) onesData[i] = 1.0f;
            ones.upload(onesData);
            biasGradient = ones.multiply(outputGradient, true, false);
        }

        // inputGradient = outputGradient * weights^T
        GpuMatrix inputGradient = outputGradient.multiply(weights, false, true);

        if (learningRate > 0) {
            weightsOpt.update(weights, weightsGradient, learningRate);
            biasOpt.update(bias, biasGradient, learningRate);
        }

        // Clean up intermediate gradients
        weightsGradient.close();
        biasGradient.close();

        return inputGradient;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        Matrix cpuW = weights.toCpu();
        Matrix cpuB = bias.toCpu();
        cpuW.save(dos);
        cpuB.save(dos);
        weightsOpt.save(dos);
        biasOpt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        if (weights != null) weights.close();
        if (bias != null) bias.close();

        Matrix cpuW = Matrix.load(dis);
        Matrix cpuB = Matrix.load(dis);
        this.weights = GpuMatrix.fromCpu(cpuW);
        this.bias = GpuMatrix.fromCpu(cpuB);

        weightsOpt.load(dis);
        biasOpt.load(dis);
    }

    @Override
    public void close() {
        if (weights != null) weights.close();
        if (bias != null) bias.close();
        if (weightsOpt != null) weightsOpt.close();
        if (biasOpt != null) biasOpt.close();
    }
}
