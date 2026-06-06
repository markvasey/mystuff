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
    public void testPositionalEncodingDimensions() {
        int seqLen = 10;
        int dModel = 32;
        PositionalEncoding pe = new PositionalEncoding(seqLen, dModel);
        Matrix input = new Matrix(seqLen, dModel);
        
        Layer.ForwardResult res = pe.forward(input);
        assertEquals(seqLen, res.output.getRows());
        assertEquals(dModel, res.output.getCols());
    }
}
