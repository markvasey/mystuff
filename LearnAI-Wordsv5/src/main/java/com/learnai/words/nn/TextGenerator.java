package com.learnai.words.nn;

import com.learnai.words.tokenizer.BPETokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class TextGenerator {
    private final OnnxLanguageModel model;
    private final BPETokenizer tokenizer;
    private final int blockSize;
    private final Random random = new Random();

    public TextGenerator(OnnxLanguageModel model, BPETokenizer tokenizer, int blockSize) {
        this.model = model;
        this.tokenizer = tokenizer;
        this.blockSize = blockSize;
    }

    public String generate(String prompt, int numTokens) {
        return generate(prompt, numTokens, 1.0, 50);
    }

    /**
     * Generates text with controlled randomness.
     * 
     * @param prompt input prompt
     * @param numTokens number of tokens to generate
     * @param temperature randomness scaling
     * @param topK candidate pool size
     * @return generated text
     */
    public String generate(String prompt, int numTokens, double temperature, int topK) {
        if (tokenizer == null) return "[Generator uninitialized]";
        
        int[] ids = tokenizer.encode(prompt);
        // During auto-regressive generation, we pass context up to blockSize tokens
        int len = Math.min(ids.length, blockSize);
        int[] currentIds = new int[len];
        System.arraycopy(ids, ids.length - len, currentIds, 0, len);

        StringBuilder result = new StringBuilder(prompt);
        for (int i = 0; i < numTokens; i++) {
            try {
                float[] logits = model.predict(currentIds);
                int nextId = sampleWithStrategy(logits, temperature, topK);
                
                result.append(tokenizer.decode(new int[]{nextId}));
                
                // Append nextId to currentIds, maintaining max length of blockSize
                if (currentIds.length < blockSize) {
                    int[] nextIds = new int[currentIds.length + 1];
                    System.arraycopy(currentIds, 0, nextIds, 0, currentIds.length);
                    nextIds[currentIds.length] = nextId;
                    currentIds = nextIds;
                } else {
                    int[] nextIds = new int[blockSize];
                    System.arraycopy(currentIds, 1, nextIds, 0, blockSize - 1);
                    nextIds[blockSize - 1] = nextId;
                    currentIds = nextIds;
                }
            } catch (Exception e) {
                System.err.println("Inference error during generation: " + e.getMessage());
                break;
            }
        }

        return result.toString();
    }

    private int sampleWithStrategy(float[] logits, double temperature, int topK) {
        int vocabSize = logits.length;
        
        // 1. Scale logits by temperature and compute softmax probabilities
        double[] probs = new double[vocabSize];
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (float val : logits) {
            if (val > maxLogit) {
                maxLogit = val;
            }
        }
        
        double sum = 0;
        for (int i = 0; i < vocabSize; i++) {
            probs[i] = Math.exp((logits[i] - maxLogit) / Math.max(temperature, 1e-6));
            sum += probs[i];
        }
        
        for (int i = 0; i < vocabSize; i++) {
            probs[i] /= sum;
        }

        // 2. Filter top-k candidates
        record TokenProb(int id, double p) {}
        List<TokenProb> candidates = new ArrayList<>();
        for (int i = 0; i < vocabSize; i++) {
            candidates.add(new TokenProb(i, probs[i]));
        }
        candidates.sort((a, b) -> Double.compare(b.p, a.p));

        int k = Math.min(topK, candidates.size());
        double topSum = 0;
        for (int i = 0; i < k; i++) {
            topSum += candidates.get(i).p;
        }

        // 3. Weighted random sampling from the top-k candidates
        double r = random.nextDouble() * topSum;
        double cumulative = 0;
        for (int i = 0; i < k; i++) {
            cumulative += candidates.get(i).p;
            if (r <= cumulative) {
                return candidates.get(i).id;
            }
        }
        
        return candidates.get(0).id;
    }
}
