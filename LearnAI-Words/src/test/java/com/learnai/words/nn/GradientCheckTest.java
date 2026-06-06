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
        Matrix inputGradAnalytical = layer.backward(outputGrad, res.context, 0.0); // lr=0 to avoid weight update

        // 2. Numerical Gradient (w.r.t input)
        double epsilon = 1e-7;
        Matrix inputGradNumerical = new Matrix(1, inDim);
        for (int i = 0; i < inDim; i++) {
            double original = input.get(0, i);
            
            input.set(0, i, original + epsilon);
            double lossPlus = computeMockLoss(layer.forward(input).output, outputGrad);
            
            input.set(0, i, original - epsilon);
            double lossMinus = computeMockLoss(layer.forward(input).output, outputGrad);
            
            input.set(0, i, original);
            inputGradNumerical.set(0, i, (lossPlus - lossMinus) / (2 * epsilon));
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
        Matrix inputGradAnalytical = layer.backward(outputGrad, res.context, 0.0);

        // 2. Numerical
        double epsilon = 1e-7;
        Matrix inputGradNumerical = new Matrix(1, dim);
        for (int i = 0; i < dim; i++) {
            double original = input.get(0, i);
            input.set(0, i, original + epsilon);
            double lossPlus = computeMockLoss(layer.forward(input).output, outputGrad);
            input.set(0, i, original - epsilon);
            double lossMinus = computeMockLoss(layer.forward(input).output, outputGrad);
            input.set(0, i, original);
            inputGradNumerical.set(0, i, (lossPlus - lossMinus) / (2 * epsilon));
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
        Matrix inputGradAnalytical = layer.backward(outputGrad, res.context, 0.0);

        // 2. Numerical
        double epsilon = 1e-7;
        Matrix inputGradNumerical = new Matrix(seqLen, dModel);
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dModel; j++) {
                double original = input.get(i, j);
                input.set(i, j, original + epsilon);
                double lossPlus = computeMockLoss(layer.forward(input).output, outputGrad);
                input.set(i, j, original - epsilon);
                double lossMinus = computeMockLoss(layer.forward(input).output, outputGrad);
                input.set(i, j, original);
                inputGradNumerical.set(i, j, (lossPlus - lossMinus) / (2 * epsilon));
            }
        }

        checkGradients(inputGradAnalytical, inputGradNumerical, "Attention Input");
    }

    private double computeMockLoss(Matrix output, Matrix outputGrad) {
        double loss = 0;
        for (int i = 0; i < output.getData().length; i++) {
            loss += output.getData()[i] * outputGrad.getData()[i];
        }
        return loss;
    }

    private void checkGradients(Matrix analytical, Matrix numerical, String name) {
        double diff = 0;
        double sum = 0;
        for (int i = 0; i < analytical.getData().length; i++) {
            double a = analytical.getData()[i];
            double n = numerical.getData()[i];
            diff += (a - n) * (a - n);
            sum += (a + n) * (a + n);
        }
        double error = Math.sqrt(diff) / Math.max(1e-7, Math.sqrt(sum));
        System.out.println(name + " Gradient Error: " + error);
        assertTrue(error < 1e-5, name + " gradient error too high: " + error);
    }
}
