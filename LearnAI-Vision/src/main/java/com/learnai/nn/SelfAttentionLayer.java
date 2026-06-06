package com.learnai.nn;

import com.learnai.math.Matrix;

/**
 * Self-Attention Layer (The heart of a Transformer).
 * Allows the network to focus on different parts of the input.
 */
public class SelfAttentionLayer implements Layer {
    private final DenseLayer queryWeights;
    private final DenseLayer keyWeights;
    private final DenseLayer valueWeights;
    private final int d_k; // Dimension of keys/queries

    private Matrix input;
    private Matrix Q, K, V;
    private Matrix attentionWeights;

    public SelfAttentionLayer(int inputDim, int headDim) {
        this.queryWeights = new DenseLayer(inputDim, headDim);
        this.keyWeights = new DenseLayer(inputDim, headDim);
        this.valueWeights = new DenseLayer(inputDim, headDim);
        this.d_k = headDim;
    }

    @Override
    public Matrix forward(Matrix input) {
        this.input = input;

        // 1. Project input into Q, K, V
        Q = queryWeights.forward(input);
        K = keyWeights.forward(input);
        V = valueWeights.forward(input);

        // 2. Calculate scores: (Q dot K^T) / sqrt(d_k)
        Matrix scores = Q.dot(K.transpose()).multiply(1.0 / Math.sqrt(d_k));

        // 3. Apply Softmax to get attention weights
        SoftmaxLayer softmax = new SoftmaxLayer();
        attentionWeights = softmax.forward(scores);

        // 4. Output: weights dot V
        return attentionWeights.dot(V);
    }

    @Override
    public Matrix backward(Matrix outputGradient, double learningRate) {
        // Backprop through Attention is mathematically heavy.
        // For our learning project, we will implement the gradients for:
        // dL/dV = Weights^T dot outputGradient
        // dL/dWeights = outputGradient dot V^T
        // ... and then backpropagate into Q, K, V linear layers.

        // Gradient w.r.t V
        Matrix dV = attentionWeights.transpose().dot(outputGradient);

        // Gradient w.r.t weights (before softmax)
        Matrix dWeights = outputGradient.dot(V.transpose());
        
        // Gradient w.r.t Q and K
        // Simplified for our first iteration:
        Matrix dQ = dWeights.dot(K).multiply(1.0 / Math.sqrt(d_k));
        Matrix dK = dWeights.transpose().dot(Q).multiply(1.0 / Math.sqrt(d_k));

        // Pass gradients back to the projection layers to update their weights
        Matrix dInputQ = queryWeights.backward(dQ, learningRate);
        Matrix dInputK = keyWeights.backward(dK, learningRate);
        Matrix dInputV = valueWeights.backward(dV, learningRate);

        // Return gradient w.r.t input (sum of contributions from Q, K, V paths)
        return dInputQ.add(dInputK).add(dInputV); 
    }
}
