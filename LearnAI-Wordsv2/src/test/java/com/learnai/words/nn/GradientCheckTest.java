package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class GradientCheckTest {

    @Test
    public void testDenseLayerGradient() {
        int inDim = 4;
        int outDim = 2;
        DenseLayer layer = new DenseLayer(inDim, outDim);
        Matrix input = Matrix.random(1, inDim);
        
        // 1. Analytical Gradient
        Layer.ForwardResult res = layer.forward(input);
        Matrix outputGrad = Matrix.random(1, outDim);
        Matrix inputGradAnalytical = layer.backward(outputGrad, res.context, 0.0f); // lr=0 to avoid weight update

        // 2. Numerical Gradient (w.r.t input)
        float epsilon = 1e-3f;
        Matrix inputGradNumerical = new Matrix(1, inDim);
        for (int i = 0; i < inDim; i++) {
            float original = input.get(0, i);
            
            input.set(0, i, original + epsilon);
            float lossPlus = computeMockLoss(layer.forward(input).output, outputGrad);
            
            input.set(0, i, original - epsilon);
            float lossMinus = computeMockLoss(layer.forward(input).output, outputGrad);
            
            input.set(0, i, original);
            inputGradNumerical.set(0, i, (lossPlus - lossMinus) / (2.0f * epsilon));
        }

        // 3. Compare
        checkGradients(inputGradAnalytical, inputGradNumerical, "DenseLayer Input");
    }

    @Test
    public void testLayerNormGradient() {
        int dim = 8;
        LayerNorm layer = new LayerNorm(dim);
        Matrix input = Matrix.random(1, dim);
        
        // 1. Analytical
        Layer.ForwardResult res = layer.forward(input);
        Matrix outputGrad = Matrix.random(1, dim);
        Matrix inputGradAnalytical = layer.backward(outputGrad, res.context, 0.0f);

        // 2. Numerical
        float epsilon = 1e-3f;
        Matrix inputGradNumerical = new Matrix(1, dim);
        for (int i = 0; i < dim; i++) {
            float original = input.get(0, i);
            input.set(0, i, original + epsilon);
            float lossPlus = computeMockLoss(layer.forward(input).output, outputGrad);
            input.set(0, i, original - epsilon);
            float lossMinus = computeMockLoss(layer.forward(input).output, outputGrad);
            input.set(0, i, original);
            inputGradNumerical.set(0, i, (lossPlus - lossMinus) / (2.0f * epsilon));
        }

        checkGradients(inputGradAnalytical, inputGradNumerical, "LayerNorm Input");
    }

    @Test
    public void testAttentionGradient() {
        int dModel = 4;
        int seqLen = 3;
        CausalSelfAttentionLayer layer = new CausalSelfAttentionLayer(dModel);
        Matrix input = Matrix.random(seqLen, dModel);

        // 1. Analytical
        Layer.ForwardResult res = layer.forward(input);
        Matrix outputGrad = Matrix.random(seqLen, dModel);
        Matrix inputGradAnalytical = layer.backward(outputGrad, res.context, 0.0f);

        // 2. Numerical
        float epsilon = 1e-3f;
        Matrix inputGradNumerical = new Matrix(seqLen, dModel);
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dModel; j++) {
                float original = input.get(i, j);
                input.set(i, j, original + epsilon);
                float lossPlus = computeMockLoss(layer.forward(input).output, outputGrad);
                input.set(i, j, original - epsilon);
                float lossMinus = computeMockLoss(layer.forward(input).output, outputGrad);
                input.set(i, j, original);
                inputGradNumerical.set(i, j, (lossPlus - lossMinus) / (2.0f * epsilon));
            }
        }

        checkGradients(inputGradAnalytical, inputGradNumerical, "Attention Input");
    }

    private float computeMockLoss(Matrix output, Matrix outputGrad) {
        float loss = 0.0f;
        for (int i = 0; i < output.getData().length; i++) {
            loss += output.getData()[i] * outputGrad.getData()[i];
        }
        return loss;
    }

    private void checkGradients(Matrix analytical, Matrix numerical, String name) {
        float diff = 0.0f;
        float sum = 0.0f;
        for (int i = 0; i < analytical.getData().length; i++) {
            float a = analytical.getData()[i];
            float n = numerical.getData()[i];
            diff += (a - n) * (a - n);
            sum += (a + n) * (a + n);
        }
        float error = (float) Math.sqrt(diff) / Math.max(1e-7f, (float) Math.sqrt(sum));
        System.out.println(name + " Gradient Error: " + error);
        assertTrue(error < 1e-3f, name + " gradient error too high: " + error);
    }
}
