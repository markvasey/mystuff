package com.learnai.words.tokenizer;

import com.learnai.words.nn.GpuLanguageModel;
import com.learnai.words.math.Matrix;
import com.learnai.words.math.GpuMatrix;
import com.learnai.words.nn.GpuEmbeddingLayer;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Scanner;

/**
 * A diagnostic tool to "peek" into the model's brain.
 * It measures the distance between word embeddings to see if the model
 * understands that certain concepts are related.
 */
public class SemanticProbe {
    public static void main(String[] args) throws Exception {
        BPETokenizer tokenizer = new BPETokenizer();
        tokenizer.load("tokenizer.bin");
        
        int dModel = Integer.getInteger("d.model", 128);
        GpuLanguageModel model = new GpuLanguageModel(tokenizer.getVocabSize(), dModel, 128);
        model.load("model.bin");

        // Use reflection to grab the embedding matrix (since it's private)
        Field embField = GpuLanguageModel.class.getDeclaredField("embedding");
        embField.setAccessible(true);
        GpuEmbeddingLayer embeddingLayer = (GpuEmbeddingLayer) embField.get(model);
        
        Field weightsField = GpuEmbeddingLayer.class.getDeclaredField("embeddings");
        weightsField.setAccessible(true);
        GpuMatrix gpuWeights = (GpuMatrix) weightsField.get(embeddingLayer);
        Matrix weights = gpuWeights.toCpu();

        Scanner scanner = new Scanner(System.in);
        System.out.println("--- Semantic Probe: Inspection Tool ---");
        System.out.println("Enter two words to see how 'close' they are in the model's mind.");
        
        while (true) {
            System.out.print("\nWord 1: ");
            String w1 = scanner.next();
            System.out.print("Word 2: ");
            String w2 = scanner.next();

            int id1 = tokenizer.encode(w1)[0];
            int id2 = tokenizer.encode(w2)[0];

            float dist = calculateEuclideanDistance(weights, id1, id2);
            System.out.printf("Distance between '%s' (ID %d) and '%s' (ID %d): %.4f\n", w1, id1, w2, id2, dist);
            System.out.println("(Lower distance = Model thinks words are more similar)");
        }
    }

    private static float calculateEuclideanDistance(Matrix weights, int id1, int id2) {
        int dim = weights.getCols();
        float sumSq = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = weights.get(id1, i) - weights.get(id2, i);
            sumSq += diff * diff;
        }
        return (float) Math.sqrt(sumSq);
    }
}
