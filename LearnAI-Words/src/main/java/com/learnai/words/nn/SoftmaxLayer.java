package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class SoftmaxLayer implements Layer {

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < input.getCols(); j++) {
                if (input.get(i, j) > max) max = input.get(i, j);
            }

            double sum = 0;
            for (int j = 0; j < input.getCols(); j++) {
                double val = Math.exp(input.get(i, j) - max);
                output.set(i, j, val);
                sum += val;
            }

            for (int j = 0; j < input.getCols(); j++) {
                output.set(i, j, output.get(i, j) / sum);
            }
        }
        return new ForwardResult(output, output);
    }

    @Override
    public Matrix backward(Matrix target, Object context, double learningRate) {
        Matrix lastOutput = (Matrix) context;
        Matrix gradient = new Matrix(lastOutput.getRows(), lastOutput.getCols());
        for (int i = 0; i < lastOutput.getRows(); i++) {
            for (int j = 0; j < lastOutput.getCols(); j++) {
                gradient.set(i, j, lastOutput.get(i, j) - target.get(i, j));
            }
        }
        return gradient;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {}

    @Override
    public void load(DataInputStream dis) throws IOException {}
}
