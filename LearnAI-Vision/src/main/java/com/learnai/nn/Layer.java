package com.learnai.nn;

import com.learnai.math.Matrix;

/**
 * Common interface for all neural network layers.
 */
public interface Layer {
    /**
     * Performs the forward pass of the layer.
     * @param input The input matrix from the previous layer.
     * @return The output matrix.
     */
    Matrix forward(Matrix input);

    /**
     * Performs the backward pass (backpropagation).
     * @param outputGradient The gradient of the loss with respect to the output of this layer.
     * @param learningRate The step size for weight updates.
     * @return The gradient of the loss with respect to the input of this layer.
     */
    Matrix backward(Matrix outputGradient, double learningRate);
}
