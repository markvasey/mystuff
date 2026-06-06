package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.CharacterTokenizer;
import com.learnai.words.tokenizer.TextDataset;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class ModelLearningTest {

    @Test
    public void testSingleSampleConvergence() {
        // Test if the model can learn a single "A -> B" relationship
        String text = "AB";
        TextDataset dataset = new TextDataset(text);
        CharacterTokenizer tokenizer = dataset.getTokenizer();
        
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), 32, 1);
        int[] input = {tokenizer.encode('A')};
        int targetId = tokenizer.encode('B');

        double initialLoss = model.train(input, targetId, 0);
        
        // Train on this single sample multiple times
        double finalLoss = 0;
        for (int i = 0; i < 50; i++) {
            finalLoss = model.train(input, targetId, 0.1);
        }

        assertTrue(finalLoss < initialLoss * 0.1, "Loss should drop significantly for a single sample. Initial: " + initialLoss + ", Final: " + finalLoss);
        
        Matrix prediction = model.predict(input);
        assertEquals(targetId, getBestIndex(prediction, 0), "Model should correctly predict B after training");
    }

    @Test
    public void testCausalMasking() {
        // Test that changing future tokens does not affect current token's prediction
        String corpus = "abcdefg";
        CharacterTokenizer tokenizer = new CharacterTokenizer(corpus);
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), 32, 5);

        int[] input1 = {tokenizer.encode('a'), tokenizer.encode('b'), tokenizer.encode('c')};
        int[] input2 = {tokenizer.encode('a'), tokenizer.encode('b'), tokenizer.encode('z')}; // Changed last token

        Matrix pred1 = model.predict(input1);
        Matrix pred2 = model.predict(input2);

        // Character at index 0 and 1 should have identical predictions in both cases
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < tokenizer.getVocabSize(); j++) {
                assertEquals(pred1.get(i, j), pred2.get(i, j), 1e-10, "Prediction at index " + i + " should be unaffected by future tokens");
            }
        }
    }

    private int getBestIndex(Matrix m, int row) {
        int best = 0;
        double max = -1;
        for (int i = 0; i < m.getCols(); i++) {
            if (m.get(row, i) > max) {
                max = m.get(row, i);
                best = i;
            }
        }
        return best;
    }
}
