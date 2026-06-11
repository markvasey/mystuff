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

        float[] originalData;
        try (GpuLanguageModel model1 = new GpuLanguageModel(vocabSize, dModel, maxLen)) {
            int[] input = {1, 2, 3, 4, 5};
            Matrix pred1 = model1.predict(input);
            originalData = pred1.getData().clone();
            model1.save(modelFile);
        }

        float[] reloadedData;
        try (GpuLanguageModel model2 = new GpuLanguageModel(vocabSize, dModel, maxLen)) {
            model2.load(modelFile);
            int[] input = {1, 2, 3, 4, 5};
            Matrix pred2 = model2.predict(input);
            reloadedData = pred2.getData().clone();
        }

        assertArrayEquals(originalData, reloadedData, 1e-6f, "Predictions should be identical after loading weights");
    }
}
