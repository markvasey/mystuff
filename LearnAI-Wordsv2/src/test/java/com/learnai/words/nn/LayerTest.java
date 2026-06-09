package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class LayerTest {

    @Test
    public void testSoftmaxProbabilities() {
        SoftmaxLayer softmax = new SoftmaxLayer();
        Matrix input = new Matrix(2, 3);
        // Row 1
        input.set(0, 0, 1.0f); input.set(0, 1, 2.0f); input.set(0, 2, 3.0f);
        // Row 2
        input.set(1, 0, 10.0f); input.set(1, 1, 10.0f); input.set(1, 2, 10.0f);

        Matrix output = softmax.forward(input).output;

        for (int i = 0; i < output.getRows(); i++) {
            float sum = 0.0f;
            for (int j = 0; j < output.getCols(); j++) {
                assertTrue(output.get(i, j) >= 0.0f && output.get(i, j) <= 1.0f);
                sum += output.get(i, j);
            }
            assertEquals(1.0f, sum, 1e-6f, "Softmax row " + i + " should sum to 1.0");
        }
    }

    @Test
    public void testEmbeddingOutput() {
        int vocabSize = 10;
        int dim = 5;
        EmbeddingLayer embedding = new EmbeddingLayer(vocabSize, dim);
        int[] input = {1, 2, 3};
        
        Matrix output = embedding.forward(input).output;
        
        assertEquals(3, output.getRows());
        assertEquals(5, output.getCols());
    }
}
