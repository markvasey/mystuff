package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class UnitLayerTest {

    @Test
    public void testDenseLayerDimensions() {
        int in = 10;
        int out = 5;
        DenseLayer layer = new DenseLayer(in, out);
        Matrix input = new Matrix(3, in); // Batch of 3
        
        Layer.ForwardResult res = layer.forward(input);
        assertEquals(3, res.output.getRows());
        assertEquals(out, res.output.getCols());
    }

    @Test
    public void testPositionalEncodingDimensionsAndMath() {
        int seqLen = 2;
        int dModel = 4;
        PositionalEncoding pe = new PositionalEncoding(seqLen, dModel);
        
        // Zero-initialized input matrix: the output should equal the PE matrix exactly
        Matrix input = new Matrix(seqLen, dModel);
        
        Layer.ForwardResult res = pe.forward(input);
        assertEquals(seqLen, res.output.getRows());
        assertEquals(dModel, res.output.getCols());

        // For pos = 0:
        // Even indices: sin(0) = 0.0
        // Odd indices: cos(0) = 1.0
        assertEquals(0.0f, res.output.get(0, 0), 1e-6f);
        assertEquals(1.0f, res.output.get(0, 1), 1e-6f);
        assertEquals(0.0f, res.output.get(0, 2), 1e-6f);
        assertEquals(1.0f, res.output.get(0, 3), 1e-6f);

        // For pos = 1:
        // Index 0: sin(1 / 10000^0) = sin(1) = 0.84147098
        // Index 1: cos(1 / 10000^0) = cos(1) = 0.5403023
        assertEquals((float) Math.sin(1.0), res.output.get(1, 0), 1e-6f);
        assertEquals((float) Math.cos(1.0), res.output.get(1, 1), 1e-6f);

        // Backward of PositionalEncoding should be identity mapping of gradient
        Matrix grad = new Matrix(seqLen, dModel);
        grad.set(0, 0, 0.5f);
        grad.set(1, 1, -0.5f);
        Matrix back = pe.backward(grad, res.context, 0.01f);
        assertEquals(0.5f, back.get(0, 0), 1e-6f);
        assertEquals(-0.5f, back.get(1, 1), 1e-6f);
    }

    @Test
    public void testResidualBlockCPU() {
        int dim = 2;
        // Mock layer that multiplies input by 2 in forward, and multiplies gradient by 2 in backward
        Layer mockSublayer = new Layer() {
            @Override
            public ForwardResult forward(Matrix input) {
                Matrix out = new Matrix(input.getRows(), input.getCols());
                for (int i = 0; i < input.getData().length; i++) {
                    out.getData()[i] = input.getData()[i] * 2.0f;
                }
                return new ForwardResult(out, null);
            }

            @Override
            public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
                Matrix grad = new Matrix(outputGradient.getRows(), outputGradient.getCols());
                for (int i = 0; i < outputGradient.getData().length; i++) {
                    grad.getData()[i] = outputGradient.getData()[i] * 2.0f;
                }
                return grad;
            }

            @Override public void save(java.io.DataOutputStream dos) {}
            @Override public void load(java.io.DataInputStream dis) {}
        };

        ResidualBlock resBlock = new ResidualBlock(mockSublayer);
        
        Matrix input = new Matrix(1, dim);
        input.set(0, 0, 1.5f);
        input.set(0, 1, -3.0f);

        // Forward: input + sublayer(input) = x + 2x = 3x
        Layer.ForwardResult forwardRes = resBlock.forward(input);
        assertEquals(4.5f, forwardRes.output.get(0, 0), 1e-6f);
        assertEquals(-9.0f, forwardRes.output.get(0, 1), 1e-6f);

        // Backward: grad + sublayer.backward(grad) = g + 2g = 3g
        Matrix grad = new Matrix(1, dim);
        grad.set(0, 0, 0.1f);
        grad.set(0, 1, 0.2f);
        Matrix back = resBlock.backward(grad, forwardRes.context, 0.01f);
        assertEquals(0.3f, back.get(0, 0), 1e-6f);
        assertEquals(0.6f, back.get(0, 1), 1e-6f);
    }
}
