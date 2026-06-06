package com.learnai.nn;

import com.learnai.math.Matrix;

/**
 * A Fully Connected (Dense) Layer.
 * Performs y = xW + b
 */
public class DenseLayer implements Layer {
    private Matrix weights;
    private Matrix biases;
    private Matrix input; // Stored for the backward pass

    public DenseLayer(int inputSize, int outputSize) {
        // Initialize weights randomly
        this.weights = Matrix.random(inputSize, outputSize);
        this.biases = new Matrix(1, outputSize); // Biases initialized to zero
    }

    @Override
    public Matrix forward(Matrix input) {
        this.input = input; // Cache input for backprop
        // Output = (Input dot Weights) + Biases
        return input.dot(weights).add(biases);
    }

    @Override
    public Matrix backward(Matrix outputGradient, double learningRate) {
        // 1. Calculate gradient with respect to weights: dL/dW = input^T * outputGradient
        Matrix weightsGradient = input.transpose().dot(outputGradient);

        // 2. Calculate gradient with respect to input (to pass back): dL/dx = outputGradient * weights^T
        Matrix inputGradient = outputGradient.dot(weights.transpose());

        // 3. Update weights and biases (Gradient Descent)
        // W = W - (learningRate * dL/dW)
        this.weights = this.weights.subtract(weightsGradient.multiply(learningRate));
        
        // b = b - (learningRate * dL/db). Sum gradients over all rows/patches for the bias update.
        Matrix biasGradient = outputGradient.sumRows();
        this.biases = this.biases.subtract(biasGradient.multiply(learningRate));

        return inputGradient;
    }
}
