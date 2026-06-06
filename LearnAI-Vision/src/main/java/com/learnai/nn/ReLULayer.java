package com.learnai.nn;

import com.learnai.math.Matrix;

/**
 * ReLU Activation Layer.
 * y = max(0, x)
 */
public class ReLULayer implements Layer {
    private Matrix input;

    @Override
    public Matrix forward(Matrix input) {
        this.input = input;
        return input.apply(x -> Math.max(0, x));
    }

    @Override
    public Matrix backward(Matrix outputGradient, double learningRate) {
        // Gradient of ReLU is 1 for x > 0, and 0 otherwise
        Matrix reluGradient = input.apply(x -> x > 0 ? 1.0 : 0.0);
        // Element-wise multiply the incoming gradient by the ReLU gradient
        return outputGradient.multiplyElementWise(reluGradient);
    }
}
