package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import com.learnai.words.tokenizer.CharacterTokenizer;
import java.util.Random;

public class TextGenerator {
    private final LanguageModel model;
    private final CharacterTokenizer tokenizer;
    private final int blockSize;

    public TextGenerator(LanguageModel model, CharacterTokenizer tokenizer, int blockSize) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
    }

    public String generate(String prompt, int length) {
        StringBuilder sb = new StringBuilder(prompt);
        Random rand = new Random();

        for (int i = 0; i < length; i++) {
            String current = sb.length() > blockSize ? 
                sb.substring(sb.length() - blockSize) : sb.toString();
            int[] ids = tokenizer.encodeString(current);
            
            Matrix probs = model.predict(ids);
            int seqLen = ids.length;
            
            int nextId = sampleFromDistribution(probs, seqLen - 1, rand);
            sb.append(tokenizer.decode(nextId));
        }
        return sb.toString();
    }

    private int sampleFromDistribution(Matrix probs, int row, Random rand) {
        double r = rand.nextDouble();
        double cumulative = 0;
        for (int j = 0; j < probs.getCols(); j++) {
            cumulative += probs.get(row, j);
            if (r <= cumulative) return j;
        }
        return probs.getCols() - 1;
    }
}
