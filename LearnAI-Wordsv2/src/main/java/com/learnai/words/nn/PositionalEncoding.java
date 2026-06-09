package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class PositionalEncoding implements Layer {
    private Matrix pe;

    public PositionalEncoding(int maxLen, int d_model) {
        this.pe = new Matrix(maxLen, d_model);
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < d_model; i++) {
                float val = (i % 2 == 0) ? 
                    (float) Math.sin(pos / Math.pow(10000, (double) i / d_model)) :
                    (float) Math.cos(pos / Math.pow(10000, (double) (i - 1) / d_model));
                pe.set(pos, i, val);
            }
        }
    }

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getCols(); j++) {
                output.set(i, j, input.get(i, j) + pe.get(i, j));
            }
        }
        return new ForwardResult(output, null);
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        return outputGradient;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        pe.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        this.pe = Matrix.load(dis);
    }
}
