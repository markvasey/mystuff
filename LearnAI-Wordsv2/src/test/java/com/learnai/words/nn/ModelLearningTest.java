package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.BPETokenizer;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ModelLearningTest {

    @Test
    public void testSingleSampleConvergence() {
        // Test if the model can learn a single "A -> B" relationship
        String text = "ABABABABABAB";
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(text, 260); // Small BPE train
        
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), 32, 1);
        int[] input = tokenizer.encode("A");
        int targetId = tokenizer.encode("B")[0];

        float initialLoss = model.train(input, targetId, 0.0f);
        
        // Train on this single sample multiple times
        float finalLoss = 0.0f;
        for (int i = 0; i < 50; i++) {
            finalLoss = model.train(input, targetId, 0.1f);
        }

        assertTrue(finalLoss < initialLoss * 0.1f, "Loss should drop significantly for a single sample. Initial: " + initialLoss + ", Final: " + finalLoss);
        
        Matrix prediction = model.predict(input);
        assertEquals(targetId, getBestIndex(prediction, 0), "Model should correctly predict B after training");
    }

    @Test
    public void testCausalMasking() {
        // Test that changing future tokens does not affect current token's prediction
        String corpus = "abcdefghijklmnopqrstuvwxyz";
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(corpus, 260);
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), 32, 5);

        int[] input1 = tokenizer.encode("abc");
        int[] input2 = tokenizer.encode("abz"); // Changed last token

        Matrix pred1 = model.predict(input1);
        Matrix pred2 = model.predict(input2);

        // First two tokens should have identical predictions in both cases
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tokenizer.getVocabSize(); j++) {
                assertEquals(pred1.get(i, j), pred2.get(i, j), 1e-5f, "Prediction at index " + i + " should be unaffected by future tokens");
            }
        }
    }

    private int getBestIndex(Matrix m, int row) {
        int best = 0;
        float max = -1.0f;
        for (int i = 0; i < m.getCols(); i++) {
            if (m.get(row, i) > max) {
                max = m.get(row, i);
                best = i;
            }
        }
        return best;
    }
}
