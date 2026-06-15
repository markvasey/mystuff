package com.learnai.words.nn;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.nio.file.Path;

public class OnnxModelTest {
    @Test
    public void testOnnxInferenceIfModelExists() throws Exception {
        Path modelPath = Path.of("model.onnx");
        if (!Files.exists(modelPath)) {
            System.out.println("model.onnx not found. Skipping ONNX inference test.");
            return;
        }

        try (OnnxLanguageModel model = new OnnxLanguageModel(modelPath.toString())) {
            int[] tokenIds = {1, 2, 3, 4};
            float[] logits = model.predict(tokenIds);
            
            assertNotNull(logits, "Model logits output should not be null");
            assertTrue(logits.length > 0, "Logits should map to vocabulary classes");
            System.out.println("ONNX model output logits size: " + logits.length);
        }
    }
}
