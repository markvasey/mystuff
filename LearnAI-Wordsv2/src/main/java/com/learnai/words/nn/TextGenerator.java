package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.BPETokenizer;
import java.util.*;

public class TextGenerator {
    private final GpuLanguageModel gpuModel;
    private final BPETokenizer tokenizer;
    private final int blockSize;
    private final Random random = new Random();

    public TextGenerator(GpuLanguageModel gpuModel, BPETokenizer tokenizer, int blockSize) {
        this.gpuModel = gpuModel;
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
    }

    public String generate(String prompt, int numTokens) {
        return generate(prompt, numTokens, 1.0, 50);
    }

    /**
     * Generates text with controlled randomness.
     * @param temperature Lower (< 1.0) = more focused/repetitive, Higher (> 1.0) = more creative/random.
     * @param topK Only consider the top K most likely tokens.
     */
    public String generate(String prompt, int numTokens, double temperature, int topK) {
        if (tokenizer == null) return "[Generator uninitialized]";
        
        int[] ids = tokenizer.encode(prompt);
        int[] currentIds = new int[blockSize];
        
        if (ids.length >= blockSize) {
            System.arraycopy(ids, ids.length - blockSize, currentIds, 0, blockSize);
        } else {
            System.arraycopy(ids, 0, currentIds, blockSize - ids.length, ids.length);
        }

        StringBuilder result = new StringBuilder(prompt);
        for (int i = 0; i < numTokens; i++) {
            Matrix probs = gpuModel.predict(currentIds);
            int nextId = sampleWithStrategy(probs, temperature, topK);
            
            result.append(tokenizer.decode(new int[]{nextId}));
            
            System.arraycopy(currentIds, 1, currentIds, 0, blockSize - 1);
            currentIds[blockSize - 1] = nextId;
        }

        return result.toString();
    }

    private int sampleWithStrategy(Matrix probs, double temperature, int topK) {
        int cols = probs.getCols();
        int lastRow = probs.getRows() - 1;
        
        // 1. Apply Temperature and create list for Top-K
        record TokenProb(int id, double p) {}
        List<TokenProb> candidates = new ArrayList<>();
        
        for (int j = 0; j < cols; j++) {
            double p = probs.get(lastRow, j);
            // Apply temperature scaling
            double adjustedP = Math.pow(p, 1.0 / Math.max(temperature, 1e-6));
            candidates.add(new TokenProb(j, adjustedP));
        }

        // 2. Sort to find the Top-K
        candidates.sort((a, b) -> Double.compare(b.p, a.p));

        // 3. Keep only Top-K
        int k = Math.min(topK, candidates.size());
        double sum = 0;
        for (int i = 0; i < k; i++) sum += candidates.get(i).p;

        // 4. Weighted random sampling from Top-K
        double r = random.nextDouble() * sum;
        double cumulative = 0;
        for (int i = 0; i < k; i++) {
            cumulative += candidates.get(i).p;
            if (r <= cumulative) return candidates.get(i).id;
        }
        
        return candidates.get(0).id;
    }
}
