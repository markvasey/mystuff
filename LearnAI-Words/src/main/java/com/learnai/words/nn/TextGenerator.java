package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.BPETokenizer;
import java.util.Random;

public class TextGenerator {
    private final LanguageModel model;
    private final BPETokenizer tokenizer;
    private final int blockSize;
    private final Random random = new Random();

    public TextGenerator(LanguageModel model, BPETokenizer tokenizer, int blockSize) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
    }

    public String generate(String prompt, int numTokens) {
        if (tokenizer == null) return "[Generator uninitialized]";
        
        int[] ids = tokenizer.encode(prompt);
        int[] currentIds = new int[blockSize];
        
        // Pad or truncate prompt to fit block size
        if (ids.length >= blockSize) {
            System.arraycopy(ids, ids.length - blockSize, currentIds, 0, blockSize);
        } else {
            System.arraycopy(ids, 0, currentIds, blockSize - ids.length, ids.length);
        }

        StringBuilder result = new StringBuilder(prompt);
        for (int i = 0; i < numTokens; i++) {
            Matrix probs = model.predict(currentIds);
            int nextId = sample(probs);
            
            result.append(tokenizer.decode(new int[]{nextId}));
            
            // Shift window
            System.arraycopy(currentIds, 1, currentIds, 0, blockSize - 1);
            currentIds[blockSize - 1] = nextId;
        }

        return result.toString();
    }

    private int sample(Matrix probs) {
        int cols = probs.getCols();
        double r = random.nextDouble();
        double cumulative = 0.0;
        for (int j = 0; j < cols; j++) {
            cumulative += probs.get(probs.getRows() - 1, j);
            if (r <= cumulative) return j;
        }
        return cols - 1;
    }
}
