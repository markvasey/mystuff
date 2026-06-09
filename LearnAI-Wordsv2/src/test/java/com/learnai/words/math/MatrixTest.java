package com.learnai.words.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {

    @Test
    public void testMultiply() {
        Matrix m1 = new Matrix(2, 3);
        m1.set(0, 0, 1.0f); m1.set(0, 1, 2.0f); m1.set(0, 2, 3.0f);
        m1.set(1, 0, 4.0f); m1.set(1, 1, 5.0f); m1.set(1, 2, 6.0f);

        Matrix m2 = new Matrix(3, 2);
        m2.set(0, 0, 7.0f); m2.set(0, 1, 8.0f);
        m2.set(1, 0, 9.0f); m2.set(1, 1, 10.0f);
        m2.set(2, 0, 11.0f); m2.set(2, 1, 12.0f);

        Matrix result = m1.multiply(m2);

        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
        assertEquals(58.0f, result.get(0, 0), 1e-6f); // 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assertEquals(64.0f, result.get(0, 1), 1e-6f); // 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assertEquals(139.0f, result.get(1, 0), 1e-6f); // 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        assertEquals(154.0f, result.get(1, 1), 1e-6f); // 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    }

    @Test
    public void testTranspose() {
        Matrix m = new Matrix(2, 3);
        m.set(0, 0, 1.0f); m.set(0, 1, 2.0f); m.set(0, 2, 3.0f);
        m.set(1, 0, 4.0f); m.set(1, 1, 5.0f); m.set(1, 2, 6.0f);

        Matrix t = m.transpose();

        assertEquals(3, t.getRows());
        assertEquals(2, t.getCols());
        assertEquals(1.0f, t.get(0, 0), 1e-6f);
        assertEquals(4.0f, t.get(0, 1), 1e-6f);
        assertEquals(2.0f, t.get(1, 0), 1e-6f);
        assertEquals(5.0f, t.get(1, 1), 1e-6f);
        assertEquals(3.0f, t.get(2, 0), 1e-6f);
        assertEquals(6.0f, t.get(2, 1), 1e-6f);
    }

    @Test
    public void testMultiplyTransposed() {
        // Test (transThis, !transOther)
        Matrix m1 = new Matrix(3, 2); // Transposed will be 2x3
        m1.set(0, 0, 1.0f); m1.set(1, 0, 2.0f); m1.set(2, 0, 3.0f);
        m1.set(0, 1, 4.0f); m1.set(1, 1, 5.0f); m1.set(2, 1, 6.0f);

        Matrix m2 = new Matrix(3, 2);
        m2.set(0, 0, 7.0f); m2.set(0, 1, 8.0f);
        m2.set(1, 0, 9.0f); m2.set(1, 1, 10.0f);
        m2.set(2, 0, 11.0f); m2.set(2, 1, 12.0f);

        Matrix result = m1.multiply(m2, true, false);

        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
        assertEquals(58.0f, result.get(0, 0), 1e-6f);
        assertEquals(64.0f, result.get(0, 1), 1e-6f);
        assertEquals(139.0f, result.get(1, 0), 1e-6f);
        assertEquals(154.0f, result.get(1, 1), 1e-6f);

        // Test (!transThis, transOther)
        Matrix m3 = new Matrix(2, 3);
        m3.set(0, 0, 1.0f); m3.set(0, 1, 2.0f); m3.set(0, 2, 3.0f);
        m3.set(1, 0, 4.0f); m3.set(1, 1, 5.0f); m3.set(1, 2, 6.0f);

        Matrix m4 = new Matrix(2, 3); // Transposed will be 3x2
        m4.set(0, 0, 7.0f); m4.set(1, 0, 8.0f);
        m4.set(0, 1, 9.0f); m4.set(1, 1, 10.0f);
        m4.set(0, 2, 11.0f); m4.set(1, 2, 12.0f);

        Matrix result2 = m3.multiply(m4, false, true);
        assertEquals(58.0f, result2.get(0, 0), 1e-6f);
    }

    @Test
    public void testAddAndSubtract() {
        Matrix m1 = new Matrix(2, 2);
        m1.set(0, 0, 1.0f); m1.set(0, 1, 2.0f);
        m1.set(1, 0, 3.0f); m1.set(1, 1, 4.0f);

        Matrix m2 = new Matrix(2, 2);
        m2.set(0, 0, 5.0f); m2.set(0, 1, 6.0f);
        m2.set(1, 0, 7.0f); m2.set(1, 1, 8.0f);

        // Test non-in-place add
        Matrix mAdd = m1.add(m2);
        assertEquals(6.0f, mAdd.get(0, 0), 1e-6f);
        assertEquals(12.0f, mAdd.get(1, 1), 1e-6f);

        // Test non-in-place subtract
        Matrix mSub = m2.subtract(m1);
        assertEquals(4.0f, mSub.get(0, 0), 1e-6f);
        assertEquals(4.0f, mSub.get(1, 1), 1e-6f);

        // Test broadcasting addInPlace
        Matrix bias = new Matrix(1, 2);
        bias.set(0, 0, 10.0f); bias.set(0, 1, 20.0f);
        m1.addInPlace(bias);
        assertEquals(11.0f, m1.get(0, 0), 1e-6f);
        assertEquals(22.0f, m1.get(0, 1), 1e-6f);
        assertEquals(13.0f, m1.get(1, 0), 1e-6f);
        assertEquals(24.0f, m1.get(1, 1), 1e-6f);
    }

    @Test
    public void testElementWiseOps() {
        Matrix m = new Matrix(1, 2);
        m.set(0, 0, 2.0f); m.set(0, 1, 3.0f);

        // square
        Matrix mSq = m.square();
        assertEquals(4.0f, mSq.get(0, 0), 1e-6f);
        assertEquals(9.0f, mSq.get(0, 1), 1e-6f);

        // sqrt
        Matrix mSqrt = mSq.sqrt(0.0f);
        assertEquals(2.0f, mSqrt.get(0, 0), 1e-6f);
        assertEquals(3.0f, mSqrt.get(0, 1), 1e-6f);

        // multiplyElementWise
        Matrix other = new Matrix(1, 2);
        other.set(0, 0, 5.0f); other.set(0, 1, 10.0f);
        Matrix mulElem = m.multiplyElementWise(other);
        assertEquals(10.0f, mulElem.get(0, 0), 1e-6f);
        assertEquals(30.0f, mulElem.get(0, 1), 1e-6f);
    }

    @Test
    public void testRowMeanAndVariance() {
        Matrix m = new Matrix(2, 3);
        // Row 0: Mean should be 2, Variance should be ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = 2/3
        m.set(0, 0, 1.0f); m.set(0, 1, 2.0f); m.set(0, 2, 3.0f);
        // Row 1: Mean should be 10, Variance should be 0
        m.set(1, 0, 10.0f); m.set(1, 1, 10.0f); m.set(1, 2, 10.0f);

        Matrix mean = m.rowMean();
        assertEquals(2.0f, mean.get(0, 0), 1e-6f);
        assertEquals(10.0f, mean.get(1, 0), 1e-6f);

        Matrix variance = m.rowVariance(mean);
        assertEquals(2.0f / 3.0f, variance.get(0, 0), 1e-6f);
        assertEquals(0.0f, variance.get(1, 0), 1e-6f);
    }
}
