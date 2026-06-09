package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.BPETokenizer;
import org.junit.jupiter.api.Test;
import java.io.*;
import static org.junit.jupiter.api.Assertions.*;

public class GpuLanguageModelTest {

    @Test
    public void testSingleSampleConvergence() {
        String text = "ABABABABABAB";
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(text, 260);
        
        try (GpuLanguageModel model = new GpuLanguageModel(tokenizer.getVocabSize(), 32, 1)) {
            int[] input = tokenizer.encode("A");
            int targetId = tokenizer.encode("B")[0];

            float initialLoss = model.train(input, targetId, 0.0f);
            
            float finalLoss = 0.0f;
            for (int i = 0; i < 50; i++) {
                finalLoss = model.train(input, targetId, 0.1f);
            }

            assertTrue(finalLoss < initialLoss * 0.1f, 
                "Loss should drop significantly for a single sample. Initial: " + initialLoss + ", Final: " + finalLoss);
            
            Matrix prediction = model.predict(input);
            assertEquals(targetId, getBestIndex(prediction, 0), "Model should correctly predict B after training");
        }
    }

    @Test
    public void testCausalMasking() {
        String corpus = "abcdefghijklmnopqrstuvwxyz";
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.train(corpus, 260);
        
        try (GpuLanguageModel model = new GpuLanguageModel(tokenizer.getVocabSize(), 32, 5)) {
            int[] input1 = tokenizer.encode("abc");
            int[] input2 = tokenizer.encode("abc"); // Using same sequence but testing prediction match
            // Wait, why was it "abc" and "abz"? Causal masking means predicting prefix doesn't change when suffix does.
            // But let's check original test structure: "abc" and "abz"
            int[] input1_orig = tokenizer.encode("abc");
            int[] input2_orig = tokenizer.encode("abz");

            Matrix pred1 = model.predict(input1_orig);
            Matrix pred2 = model.predict(input2_orig);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < tokenizer.getVocabSize(); j++) {
                    assertEquals(pred1.get(i, j), pred2.get(i, j), 1e-5f, 
                        "Prediction at index " + i + " should be unaffected by future tokens");
                }
            }
        }
    }

    @Test
    public void testGpuModelMatchesCpu() throws Exception {
        int vocabSize = 10;
        int d_model = 8;
        int maxLen = 4;
        
        LanguageModel cpuModel = new LanguageModel(vocabSize, d_model, maxLen);
        
        // Sync parameters
        File tempFile = File.createTempFile("model_sync", ".bin");
        tempFile.deleteOnExit();
        cpuModel.save(tempFile.getAbsolutePath());
        
        GpuLanguageModel gpuModel = new GpuLanguageModel(vocabSize, d_model, maxLen);
        gpuModel.load(tempFile.getAbsolutePath());
        tempFile.delete();
        
        // Verify predict matches
        int[] input = {1, 3, 0, 7};
        Matrix cpuPred = cpuModel.predict(input);
        Matrix gpuPred = gpuModel.predict(input);
        
        assertArrayEquals(cpuPred.getData(), gpuPred.getData(), 1e-5f);
        
        // Verify training updates match
        int targetId = 5;
        float lr = 0.05f;
        
        float cpuLoss = cpuModel.train(input, targetId, lr);
        float gpuLoss = gpuModel.train(input, targetId, lr);
        
        assertEquals(cpuLoss, gpuLoss, 1e-5f);
        
        // Save both and compare serializations
        File tempCpu = File.createTempFile("model_cpu", ".bin");
        tempCpu.deleteOnExit();
        cpuModel.save(tempCpu.getAbsolutePath());
        
        File tempGpu = File.createTempFile("model_gpu", ".bin");
        tempGpu.deleteOnExit();
        gpuModel.save(tempGpu.getAbsolutePath());
        
        // Read matrices and compare values directly
        try (DataInputStream disCpu2 = new DataInputStream(new FileInputStream(tempCpu));
             DataInputStream disGpu2 = new DataInputStream(new FileInputStream(tempGpu))) {
            
            // Epochs
            assertEquals(disCpu2.readInt(), disGpu2.readInt());
            
            // Embedding layer weights
            assertArrayEquals(Matrix.load(disCpu2).getData(), Matrix.load(disGpu2).getData(), 1e-5f);
            // Embedding optimizer states
            assertArrayEquals(Matrix.load(disCpu2).getData(), Matrix.load(disGpu2).getData(), 1e-5f);
            assertArrayEquals(Matrix.load(disCpu2).getData(), Matrix.load(disGpu2).getData(), 1e-5f);
            assertEquals(disCpu2.readInt(), disGpu2.readInt());
            
            // Positional encoding weights
            assertArrayEquals(Matrix.load(disCpu2).getData(), Matrix.load(disGpu2).getData(), 1e-5f);
            
            // Sequential layers count
            int cpuLayersCount = disCpu2.readInt();
            int gpuLayersCount = disGpu2.readInt();
            assertEquals(cpuLayersCount, gpuLayersCount);
            
            // Compare each layer in structured sequence
            for (int i = 0; i < 3; i++) {
                assertLayerNormEquals(disCpu2, disGpu2);
                assertAttentionEquals(disCpu2, disGpu2);
                assertLayerNormEquals(disCpu2, disGpu2);
                assertDenseEquals(disCpu2, disGpu2);
            }
            assertLayerNormEquals(disCpu2, disGpu2);
            assertDenseEquals(disCpu2, disGpu2);
            
        } finally {
            tempCpu.delete();
            tempGpu.delete();
            gpuModel.close();
        }
    }

    private void assertAdamEquals(DataInputStream disCpu, DataInputStream disGpu) throws IOException {
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertEquals(disCpu.readInt(), disGpu.readInt());
    }

    private void assertLayerNormEquals(DataInputStream disCpu, DataInputStream disGpu) throws IOException {
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertAdamEquals(disCpu, disGpu);
        assertAdamEquals(disCpu, disGpu);
    }

    private void assertAttentionEquals(DataInputStream disCpu, DataInputStream disGpu) throws IOException {
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertAdamEquals(disCpu, disGpu);
        assertAdamEquals(disCpu, disGpu);
        assertAdamEquals(disCpu, disGpu);
    }

    private void assertDenseEquals(DataInputStream disCpu, DataInputStream disGpu) throws IOException {
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertArrayEquals(Matrix.load(disCpu).getData(), Matrix.load(disGpu).getData(), 1e-5f);
        assertAdamEquals(disCpu, disGpu);
        assertAdamEquals(disCpu, disGpu);
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
