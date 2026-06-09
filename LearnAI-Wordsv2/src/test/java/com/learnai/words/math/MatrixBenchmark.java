package com.learnai.words.math;

import org.junit.jupiter.api.Test;

public class MatrixBenchmark {

    @Test
    public void runBenchmarks() {
        int rows = 128;
        int cols = 64;
        int inner = 128;
        int iterations = 10000;

        Matrix a = Matrix.random(rows, inner);
        Matrix b = Matrix.random(cols, inner); // b is [64 x 128], so b^T is [128 x 64]
        
        System.out.println("--- Starting Matrix Benchmarks (" + iterations + " iterations) ---");

        // 1. Transpose + Multiply (Legacy)
        long start = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            Matrix bt = b.transpose();
            Matrix res = a.multiply(bt);
        }
        long end = System.currentTimeMillis();
        System.out.println("Legacy (Transpose + Multiply): " + (end - start) + "ms");

        // 2. Transpose-Free Multiply (Optimized)
        start = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            Matrix res = a.multiply(b, false, true);
        }
        end = System.currentTimeMillis();
        System.out.println("Optimized (Transpose-Free): " + (end - start) + "ms");

        // 3. In-Place Addition
        Matrix c = Matrix.random(rows, cols);
        Matrix d = Matrix.random(rows, cols);
        start = System.currentTimeMillis();
        for (int i = 0; i < iterations; i++) {
            c.addInPlace(d);
        }
        end = System.currentTimeMillis();
        System.out.println("In-Place Addition: " + (end - start) + "ms");
        
        System.out.println("--- Benchmarks Complete ---");
    }
}
