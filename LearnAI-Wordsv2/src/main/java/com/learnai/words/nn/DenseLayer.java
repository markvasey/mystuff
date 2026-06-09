package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class DenseLayer implements Layer {
    private Matrix weights;
    private Matrix bias;
    private final Adam weightsOpt;
    private final Adam biasOpt;

    public DenseLayer(int inputDim, int outputDim) {
        this.weights = Matrix.random(inputDim, outputDim);
        this.bias = new Matrix(1, outputDim); // new Matrix is zero-initialized
        this.weightsOpt = new Adam(inputDim, outputDim);
        this.biasOpt = new Adam(1, outputDim);
    }

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix output = input.multiply(weights).add(bias);
        return new ForwardResult(output, input);
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        Matrix lastInput = (Matrix) context;
        
        // weightsGradient = lastInput^T * outputGradient
        Matrix weightsGradient = lastInput.multiply(outputGradient, true, false);
        
        // biasGradient = sum of outputGradient along the row dimension (sequence dimension)
        Matrix biasGradient = new Matrix(1, bias.getCols());
        float[] bg = biasGradient.getData();
        float[] og = outputGradient.getData();
        int ogRows = outputGradient.getRows();
        int ogCols = outputGradient.getCols();
        
        for (int i = 0; i < ogRows; i++) {
            int off = i * ogCols;
            for (int j = 0; j < ogCols; j++) {
                bg[j] += og[off + j];
            }
        }

        // inputGradient = outputGradient * weights^T
        Matrix inputGradient = outputGradient.multiply(weights, false, true);

        if (learningRate > 0) {
            weightsOpt.update(weights, weightsGradient, learningRate);
            biasOpt.update(bias, biasGradient, learningRate);
        }

        return inputGradient;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        weights.save(dos);
        bias.save(dos);
        weightsOpt.save(dos);
        biasOpt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        this.weights = Matrix.load(dis);
        this.bias = Matrix.load(dis);
        this.weightsOpt.load(dis);
        this.biasOpt.load(dis);
    }
}
