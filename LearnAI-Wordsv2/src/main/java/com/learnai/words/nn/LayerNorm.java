package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class LayerNorm implements Layer {
    private Matrix gamma; // Gain
    private Matrix beta;  // Bias
    private final Adam gammaOpt;
    private final Adam betaOpt;
    private final float eps = 1e-5f;

    private static class LNState {
        public final Matrix input;
        public final Matrix xHat;
        public final Matrix var;
        public LNState(Matrix i, Matrix x, Matrix v) {
            this.input = i;
            this.xHat = x;
            this.var = v;
        }
    }

    public LayerNorm(int dim) {
        this.gamma = new Matrix(1, dim);
        // Initialize gains to 1.0
        for (int i = 0; i < dim; i++) this.gamma.set(0, i, 1.0f);
        this.beta = new Matrix(1, dim); // Zero initialized
        this.gammaOpt = new Adam(1, dim);
        this.betaOpt = new Adam(1, dim);
    }

    @Override
    public ForwardResult forward(Matrix input) {
        int N = input.getRows();
        int D = input.getCols();
        
        Matrix mean = input.rowMean();
        Matrix var = input.rowVariance(mean);
        
        Matrix xHat = new Matrix(N, D);
        Matrix output = new Matrix(N, D);
        
        for (int i = 0; i < N; i++) {
            float m = mean.get(i, 0);
            float v = var.get(i, 0);
            float invStd = (float) (1.0 / Math.sqrt(v + eps));
            for (int j = 0; j < D; j++) {
                float xh = (input.get(i, j) - m) * invStd;
                xHat.set(i, j, xh);
                output.set(i, j, xh * gamma.get(0, j) + beta.get(0, j));
            }
        }
        
        return new ForwardResult(output, new LNState(input, xHat, var));
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        LNState state = (LNState) context;
        int N = outputGradient.getRows();
        int D = outputGradient.getCols();

        Matrix dGamma = new Matrix(1, D);
        Matrix dBeta = new Matrix(1, D);
        Matrix dXHat = new Matrix(N, D);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                float outGrad = outputGradient.get(i, j);
                dGamma.set(0, j, dGamma.get(0, j) + outGrad * state.xHat.get(i, j));
                dBeta.set(0, j, dBeta.get(0, j) + outGrad);
                dXHat.set(i, j, outGrad * gamma.get(0, j));
            }
        }

        Matrix dInput = new Matrix(N, D);
        Matrix dXHat_rowMean = dXHat.rowMean();
        Matrix xHat_dXHat_rowMean = dXHat.multiplyElementWise(state.xHat).rowMean();

        for (int i = 0; i < N; i++) {
            float v = state.var.get(i, 0);
            float invStd = (float) (1.0 / Math.sqrt(v + eps));
            float dxh_rm = dXHat_rowMean.get(i, 0);
            float xh_dxh_rm = xHat_dXHat_rowMean.get(i, 0);
            
            for (int j = 0; j < D; j++) {
                float val = invStd * (dXHat.get(i, j) - dxh_rm - state.xHat.get(i, j) * xh_dxh_rm);
                dInput.set(i, j, val);
            }
        }

        if (learningRate > 0) {
            gammaOpt.update(gamma, dGamma, learningRate);
            betaOpt.update(beta, dBeta, learningRate);
        }

        return dInput;
    }

    @Override public void save(DataOutputStream dos) throws IOException { gamma.save(dos); beta.save(dos); gammaOpt.save(dos); betaOpt.save(dos); }
    @Override public void load(DataInputStream dis) throws IOException { gamma = Matrix.load(dis); beta = Matrix.load(dis); gammaOpt.load(dis); betaOpt.load(dis); }
}
