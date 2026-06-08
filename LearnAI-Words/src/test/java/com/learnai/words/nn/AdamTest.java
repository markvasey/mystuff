package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class AdamTest {

    @Test
    public void testAdamUpdate() {
        // Initialize weights with 1.0, gradient with 0.5
        Matrix weights = new Matrix(1, 1);
        weights.set(0, 0, 1.0);
        Matrix gradient = new Matrix(1, 1);
        gradient.set(0, 0, 0.5);

        Adam opt = new Adam(1, 1);
        
        // Run first update step with lr = 0.1
        // For step 1:
        // m_1 = 0.9 * 0 + (1-0.9) * 0.5 = 0.05
        // v_1 = 0.999 * 0 + (1-0.999) * 0.25 = 0.00025
        // m_hat = 0.05 / (1 - 0.9^1) = 0.05 / 0.1 = 0.5
        // v_hat = 0.00025 / (1 - 0.999^1) = 0.00025 / 0.001 = 0.25
        // delta_w = -0.1 * 0.5 / (sqrt(0.25) + 1e-8) = -0.1 * 0.5 / 0.5 = -0.1
        // w_new = 1.0 - 0.1 = 0.9
        opt.update(weights, gradient, 0.1);
        assertEquals(0.9, weights.get(0, 0), 1e-6);
    }
}
