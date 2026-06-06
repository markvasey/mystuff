package com.learnai.nn;

import com.learnai.math.Matrix;

/**
 * Averages all rows in a matrix into a single row.
 * Used to summarize patch information before final classification.
 */
public class GlobalAveragePoolingLayer implements Layer {
    private int originalRows;

    @Override
    public Matrix forward(Matrix input) {
        this.originalRows = input.getRows();
        Matrix result = new Matrix(1, input.getCols());
        for (int j = 0; j < input.getCols(); j++) {
            double sum = 0;
            for (int i = 0; i < input.getRows(); i++) {
                sum += input.get(i, j);
            }
            result.set(0, j, sum / originalRows);
        }
        return result;
    }

    @Override
    public Matrix backward(Matrix outputGradient, double learningRate) {
        // Gradient is just the output gradient distributed across all rows
        Matrix inputGradient = new Matrix(originalRows, outputGradient.getCols());
        for (int i = 0; i < originalRows; i++) {
            for (int j = 0; j < outputGradient.getCols(); j++) {
                inputGradient.set(i, j, outputGradient.get(0, j) / originalRows);
            }
        }
        return inputGradient;
    }
}
