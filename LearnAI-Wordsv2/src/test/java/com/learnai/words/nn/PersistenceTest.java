package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.CharacterTokenizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class PersistenceTest {

    @TempDir
    Path tempDir;

    @Test
    public void testSaveLoadConsistency() throws IOException {
        int vocabSize = 50;
        int dModel = 32;
        int maxLen = 16;
        String modelFile = tempDir.resolve("test_model.bin").toString();

        // 1. Create and initialize a model with random weights
        LanguageModel model1 = new LanguageModel(vocabSize, dModel, maxLen);
        
        // 2. Capture a prediction from the original model
        int[] input = {1, 2, 3, 4, 5};
        Matrix pred1 = model1.predict(input);
        float[] originalData = pred1.getData().clone();

        // 3. Save the model
        model1.save(modelFile);

        // 4. Create a new model and load the weights
        LanguageModel model2 = new LanguageModel(vocabSize, dModel, maxLen);
        model2.load(modelFile);

        // 5. Capture a prediction from the reloaded model
        Matrix pred2 = model2.predict(input);
        float[] reloadedData = pred2.getData();

        // 6. Assert that the outputs are identical
        assertArrayEquals(originalData, reloadedData, 1e-6f, "Predictions should be identical after loading weights");
    }
}
