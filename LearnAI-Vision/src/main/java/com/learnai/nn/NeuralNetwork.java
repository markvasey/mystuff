package com.learnai.nn;

import com.learnai.math.Matrix;
import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates a sequence of layers to form a complete network.
 */
public class NeuralNetwork {
    private final List<Layer> layers = new ArrayList<>();

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public Matrix predict(Matrix input) {
        Matrix output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public void train(Matrix input, Matrix target, double learningRate) {
        // 1. Forward Pass
        Matrix output = predict(input);

        // 2. Calculate initial error (dL/dy)
        // For Cross Entropy + Softmax, the starting gradient is (predictions - targets)
        Matrix gradient = output.subtract(target);

        // 3. Backward Pass (through layers in reverse)
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient, learningRate);
        }
    }

    public double calculateLoss(Matrix prediction, Matrix target) {
        // Cross Entropy Loss: -sum(target * log(prediction))
        double loss = 0;
        for (int i = 0; i < prediction.getRows(); i++) {
            for (int j = 0; j < prediction.getCols(); j++) {
                if (target.get(i, j) > 0) {
                    loss -= Math.log(Math.max(prediction.get(i, j), 1e-15));
                }
            }
        }
        return loss / prediction.getRows();
    }
}
