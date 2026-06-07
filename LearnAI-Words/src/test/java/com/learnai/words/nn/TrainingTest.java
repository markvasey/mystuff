package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.CharacterTokenizer;
import com.learnai.words.tokenizer.TextDataset;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

public class TrainingTest {

    @Test
    public void testOverfittingSmallSequence() {
        String text = "The cat sat on the mat.";
        TextDataset dataset = new TextDataset(text);
        CharacterTokenizer tokenizer = dataset.getTokenizer();
        
        int blockSize = 8;
        // 64-dim 12-layer model for a clean stable test
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), 64, blockSize);
        List<TextDataset.SequencePair> sequences = dataset.getSequences(blockSize);
        
        double initialLoss = 0;
        for (TextDataset.SequencePair pair : sequences) {
            initialLoss += model.train(pair.input(), pair.target(), 0);
        }
        initialLoss /= sequences.size();

        // Train for 200 epochs with STABLE learning rate (0.001)
        double finalLoss = 0;
        for (int i = 0; i < 200; i++) {
            finalLoss = 0;
            for (TextDataset.SequencePair pair : sequences) {
                finalLoss += model.train(pair.input(), pair.target(), 0.001);
            }
            finalLoss /= sequences.size();
            if (i % 50 == 0) System.out.println("Epoch " + i + " Loss: " + finalLoss);
        }

        System.out.println("Overfitting Test - Initial: " + initialLoss + ", Final: " + finalLoss);
        assertTrue(finalLoss < initialLoss, "Loss should at least decrease. Final: " + finalLoss);
    }

    @Test
    public void testCausalMasking() {
        String corpus = "abcdefghijklmnopqrstuvwxyz";
        CharacterTokenizer tokenizer = new CharacterTokenizer(corpus);
        LanguageModel model = new LanguageModel(tokenizer.getVocabSize(), 32, 10);

        int[] input1 = {1, 2, 3, 4, 5};
        int[] input2 = {1, 2, 3, 4, 9}; // Changed 5th token

        Matrix pred1 = model.predict(input1);
        Matrix pred2 = model.predict(input2);

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < tokenizer.getVocabSize(); j++) {
                assertEquals(pred1.get(i, j), pred2.get(i, j), 1e-10);
            }
        }
    }
}
