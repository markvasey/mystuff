package com.learnai;

import com.learnai.math.Matrix;
import com.learnai.nn.*;
import java.io.File;

public class TestRunner {
    public static void main(String[] args) {
        System.out.println("--- Starting Unit Tests ---");
        
        try {
            testMatrixMath();
            System.out.println("✓ Matrix Math Passed");
            
            testDenseLayer();
            System.out.println("✓ Dense Layer Passed");
            
            testReLU();
            System.out.println("✓ ReLU Layer Passed");
            
            testSoftmax();
            System.out.println("✓ Softmax Layer Passed");
            
            testGlobalPooling();
            System.out.println("✓ Global Average Pooling Passed");
            
            System.out.println("\n--- All Tests Passed Successfully! ---");
        } catch (Throwable e) {
            System.err.println("\n✖ Test Failed!");
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void testMatrixMath() {
        Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix m2 = new Matrix(new double[][]{{5, 6}, {7, 8}});
        
        // Addition
        Matrix sum = m1.add(m2);
        assert sum.get(0, 0) == 6;
        assert sum.get(1, 1) == 12;

        // Dot Product
        Matrix dot = m1.dot(m2);
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert dot.get(0, 0) == 19;
        assert dot.get(1, 1) == 50;

        // Broadcasting
        Matrix row = new Matrix(new double[][]{{10, 10}});
        Matrix broadcasted = m1.add(row);
        assert broadcasted.get(0, 0) == 11;
        assert broadcasted.get(1, 0) == 13;
    }

    private static void testDenseLayer() {
        DenseLayer layer = new DenseLayer(2, 3);
        Matrix input = new Matrix(new double[][]{{1, 0}});
        Matrix output = layer.forward(input);
        assert output.getRows() == 1;
        assert output.getCols() == 3;
        
        // Backward pass should not crash
        Matrix grad = new Matrix(new double[][]{{0.1, 0.1, 0.1}});
        Matrix inGrad = layer.backward(grad, 0.01);
        assert inGrad.getRows() == 1;
        assert inGrad.getCols() == 2;
    }

    private static void testReLU() {
        ReLULayer relu = new ReLULayer();
        Matrix input = new Matrix(new double[][]{{-1, 2}, {0.5, -3}});
        Matrix output = relu.forward(input);
        assert output.get(0, 0) == 0;
        assert output.get(0, 1) == 2;
        assert output.get(1, 0) == 0.5;
        assert output.get(1, 1) == 0;
    }

    private static void testSoftmax() {
        SoftmaxLayer softmax = new SoftmaxLayer();
        Matrix input = new Matrix(new double[][]{{1, 1, 1}}); // Equal scores
        Matrix output = softmax.forward(input);
        for (int j = 0; j < 3; j++) {
            assert Math.abs(output.get(0, j) - 0.3333) < 0.01;
        }
        assert Math.abs(output.sum() - 1.0) < 1e-9;
    }

    private static void testGlobalPooling() {
        GlobalAveragePoolingLayer pool = new GlobalAveragePoolingLayer();
        Matrix input = new Matrix(new double[][]{{1, 2}, {3, 4}, {5, 6}});
        Matrix output = pool.forward(input);
        assert output.getRows() == 1;
        // (1+3+5)/3 = 3, (2+4+6)/3 = 4
        assert output.get(0, 0) == 3.0;
        assert output.get(0, 1) == 4.0;
    }
}
