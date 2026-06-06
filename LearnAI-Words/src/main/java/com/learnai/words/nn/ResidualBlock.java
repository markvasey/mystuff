package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class ResidualBlock implements Layer {
    private final Layer sublayer;

    public ResidualBlock(Layer sublayer) {
        this.sublayer = sublayer;
    }

    @Override
    public ForwardResult forward(Matrix input) {
        ForwardResult res = sublayer.forward(input);
        // x = x + sublayer(x)
        Matrix output = res.output.add(input);
        return new ForwardResult(output, res.context);
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, double learningRate) {
        // Gradient of (x + f(x)) w.r.t input is (1 + f'(x))
        Matrix sublayerGrad = sublayer.backward(outputGradient, context, learningRate);
        return sublayerGrad.add(outputGradient);
    }

    @Override public void save(DataOutputStream dos) throws IOException { sublayer.save(dos); }
    @Override public void load(DataInputStream dis) throws IOException { sublayer.load(dis); }
}
