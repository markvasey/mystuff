package com.learnai.cli;

import com.learnai.math.Matrix;
import com.learnai.nn.*;
import com.learnai.vision.ImageProcessor;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import java.util.Arrays;
import java.util.concurrent.atomic.DoubleAdder;

public class VisionCLI {
    private static final String[] FAMILY = {"emily", "mark", "maya", "nikki", "ollie", "sam"};
    private static final int IMAGE_SIZE = 32;
    private static final int PATCH_SIZE = 8;
    private static final int HIDDEN_DIM = 64;

    public static void main(String[] args) throws IOException {
        System.out.println("--- Family Vision Transformer Training (Multi-threaded) ---");

        ImageProcessor processor = new ImageProcessor(IMAGE_SIZE, PATCH_SIZE);
        NeuralNetwork model = buildModel();

        File trainingDir = new File("Training");
        File[] files = trainingDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".jpg") || name.toLowerCase().endsWith(".jpeg"));

        if (files == null || files.length == 0) {
            System.out.println("No training images found in Training/ directory!");
            return;
        }

        System.out.println("Found " + files.length + " images. Utilizing available CPUs for training...");

        double learningRate = 0.005;
        int epochs = 500;

        List<File> fileList = Arrays.asList(files);

        for (int epoch = 1; epoch <= epochs; epoch++) {
            DoubleAdder totalLoss = new DoubleAdder();
            
            // Process images in parallel across all available cores
            fileList.parallelStream().forEach(file -> {
                try {
                    // 1. Process image (Thread-safe)
                    Matrix patches = processor.imageToPatches(file);
                    Matrix target = getTarget(file.getName());
                    
                    // 2. Predict and Train
                    // We synchronize the model update to prevent race conditions 
                    // while multiple threads calculate gradients.
                    synchronized (model) {
                        Matrix prediction = model.predict(patches);
                        totalLoss.add(model.calculateLoss(prediction, target));
                        model.train(patches, target, learningRate);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            });

            if (epoch % 10 == 0) {
                System.out.printf("Epoch %d/%d - Average Loss: %.4f\n", epoch, epochs, totalLoss.doubleValue() / files.length);
            }
        }

        System.out.println("\nTraining Complete! Let's test the model on the training set:");
        for (File file : files) {
            Matrix patches = processor.imageToPatches(file);
            Matrix prediction = model.predict(patches);
            int predictedIdx = getBestIndex(prediction);
            System.out.println("File: " + file.getName() + " | Predicted: " + FAMILY[predictedIdx]);
        }
    }

    private static NeuralNetwork buildModel() {
        NeuralNetwork nn = new NeuralNetwork();
        int patchDim = PATCH_SIZE * PATCH_SIZE * 3;

        // 1. Patch Embedding: (NumPatches, patchDim) -> (NumPatches, HIDDEN_DIM)
        nn.addLayer(new DenseLayer(patchDim, HIDDEN_DIM));
        
        // 2. Self Attention Layer
        nn.addLayer(new SelfAttentionLayer(HIDDEN_DIM, HIDDEN_DIM));
        
        // 3. Activation
        nn.addLayer(new ReLULayer());
        
        // 4. Global Average Pooling (Summarize patches)
        nn.addLayer(new GlobalAveragePoolingLayer());
        
        // 5. Classification Head
        nn.addLayer(new DenseLayer(HIDDEN_DIM, FAMILY.length));
        
        // 6. Softmax
        nn.addLayer(new SoftmaxLayer());
        
        return nn;
    }

    private static Matrix getTarget(String filename) {
        String name = filename.toLowerCase();
        Matrix target = new Matrix(1, FAMILY.length);
        for (int i = 0; i < FAMILY.length; i++) {
            if (name.contains(FAMILY[i])) {
                target.set(0, i, 1.0);
                return target;
            }
        }
        return target;
    }

    private static int getBestIndex(Matrix m) {
        int best = 0;
        double max = -1;
        for (int i = 0; i < m.getCols(); i++) {
            if (m.get(0, i) > max) {
                max = m.get(0, i);
                best = i;
            }
        }
        return best;
    }
}
