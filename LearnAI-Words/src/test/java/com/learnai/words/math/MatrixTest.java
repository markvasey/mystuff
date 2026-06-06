package com.learnai.words.math;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class MatrixTest {

    @Test
    public void testMultiply() {
        Matrix m1 = new Matrix(2, 3);
        m1.set(0, 0, 1); m1.set(0, 1, 2); m1.set(0, 2, 3);
        m1.set(1, 0, 4); m1.set(1, 1, 5); m1.set(1, 2, 6);

        Matrix m2 = new Matrix(3, 2);
        m2.set(0, 0, 7); m2.set(0, 1, 8);
        m2.set(1, 0, 9); m2.set(1, 1, 10);
        m2.set(2, 0, 11); m2.set(2, 1, 12);

        Matrix result = m1.multiply(m2);

        assertEquals(2, result.getRows());
        assertEquals(2, result.getCols());
        assertEquals(58, result.get(0, 0)); // 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        assertEquals(64, result.get(0, 1)); // 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        assertEquals(139, result.get(1, 0)); // 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        assertEquals(154, result.get(1, 1)); // 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    }

    @Test
    public void testTranspose() {
        Matrix m = new Matrix(2, 3);
        m.set(0, 0, 1); m.set(0, 1, 2); m.set(0, 2, 3);
        m.set(1, 0, 4); m.set(1, 1, 5); m.set(1, 2, 6);

        Matrix t = m.transpose();

        assertEquals(3, t.getRows());
        assertEquals(2, t.getCols());
        assertEquals(1, t.get(0, 0));
        assertEquals(4, t.get(0, 1));
        assertEquals(2, t.get(1, 0));
        assertEquals(5, t.get(1, 1));
        assertEquals(3, t.get(2, 0));
        assertEquals(6, t.get(2, 1));
    }

    @Test
    public void testAddBroadcasting() {
        Matrix m = new Matrix(2, 2);
        m.set(0, 0, 1); m.set(0, 1, 2);
        m.set(1, 0, 3); m.set(1, 1, 4);

        Matrix bias = new Matrix(1, 2);
        bias.set(0, 0, 10); bias.set(0, 1, 20);

        Matrix result = m.add(bias);

        assertEquals(11, result.get(0, 0)); // 1 + 10
        assertEquals(22, result.get(0, 1)); // 2 + 20
        assertEquals(13, result.get(1, 0)); // 3 + 10
        assertEquals(24, result.get(1, 1)); // 4 + 20
    }
}
