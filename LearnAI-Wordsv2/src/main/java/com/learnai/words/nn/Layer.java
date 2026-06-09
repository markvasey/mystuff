package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public interface Layer {
    /**
     * Data class to hold activations for the backward pass.
     */
    class ForwardResult {
        public final Matrix output;
        public final Object context;

        public ForwardResult(Matrix output, Object context) {
            this.output = output;
            this.context = context;
        }
    }

    ForwardResult forward(Matrix input);
    
    /**
     * @return gradient with respect to input
     */
    Matrix backward(Matrix outputGradient, Object context, float learningRate);
    
    void save(DataOutputStream dos) throws IOException;
    void load(DataInputStream dis) throws IOException;
}
