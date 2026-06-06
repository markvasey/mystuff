package com.learnai.nn;

import com.learnai.math.Matrix;

/**
 * Softmax Activation Layer.
 * Turns raw scores into probabilities.
 */
public class SoftmaxLayer implements Layer {
    private Matrix output;

    @Override
    public Matrix forward(Matrix input) {
        Matrix result = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            double sum = 0;
            double max = Double.NEGATIVE_INFINITY;
            
            // Find max for numerical stability (preventing exp() from exploding)
            for (int j = 0; j < input.getCols(); j++) {
                max = Math.max(max, input.get(i, j));
            }

            for (int j = 0; j < input.getCols(); j++) {
                double val = Math.exp(input.get(i, j) - max);
                result.set(i, j, val);
                sum += val;
            }
            
            for (int j = 0; j < input.getCols(); j++) {
                result.set(i, j, result.get(i, j) / sum);
            }
        }
        this.output = result;
        return result;
    }

    @Override
    public Matrix backward(Matrix outputGradient, double learningRate) {
        // Combined Softmax + CrossEntropy backward is usually simplified to (pred - target).
        // For a standalone layer, it's more complex, but we'll use the simplified version
        // in our training loop for now.
        return outputGradient; 
    }
}
