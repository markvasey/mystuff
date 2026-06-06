package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class LayerNorm implements Layer {
    private Matrix gamma;
    private Matrix beta;
    private final double eps = 1e-5;
    private final Adam gammaOpt;
    private final Adam betaOpt;

    private static class LNState {
        public final Matrix input;
        public final Matrix mean;
        public final Matrix var;
        public final Matrix xHat;
        public LNState(Matrix input, Matrix mean, Matrix var, Matrix xHat) {
            this.input = input; this.mean = mean; this.var = var; this.xHat = xHat;
        }
    }

    public LayerNorm(int dim) {
        this.gamma = new Matrix(1, dim);
        for (int i = 0; i < dim; i++) gamma.set(0, i, 1.0);
        this.beta = new Matrix(1, dim);
        this.gammaOpt = new Adam(1, dim);
        this.betaOpt = new Adam(1, dim);
    }

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix mean = input.rowMean();
        Matrix var = input.rowVariance(mean);
        Matrix xHat = new Matrix(input.getRows(), input.getCols());
        Matrix output = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            double m = mean.get(i, 0);
            double v = var.get(i, 0);
            double invStd = 1.0 / Math.sqrt(v + eps);
            for (int j = 0; j < input.getCols(); j++) {
                double xh = (input.get(i, j) - m) * invStd;
                xHat.set(i, j, xh);
                output.set(i, j, xh * gamma.get(0, j) + beta.get(0, j));
            }
        }
        return new ForwardResult(output, new LNState(input, mean, var, xHat));
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, double learningRate) {
        LNState state = (LNState) context;
        int N = outputGradient.getRows();
        int D = outputGradient.getCols();
        Matrix dGamma = new Matrix(1, D);
        Matrix dBeta = new Matrix(1, D);
        Matrix dXHat = new Matrix(N, D);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < D; j++) {
                dGamma.set(0, j, dGamma.get(0, j) + outputGradient.get(i, j) * state.xHat.get(i, j));
                dBeta.set(0, j, dBeta.get(0, j) + outputGradient.get(i, j));
                dXHat.set(i, j, outputGradient.get(i, j) * gamma.get(0, j));
            }
        }
        Matrix dInput = new Matrix(N, D);
        Matrix dXHat_rowMean = dXHat.rowMean();
        Matrix xHat_dXHat_rowMean = dXHat.multiplyElementWise(state.xHat).rowMean();

        for (int i = 0; i < N; i++) {
            double v = state.var.get(i, 0);
            double invStd = 1.0 / Math.sqrt(v + eps);
            double dxh_rm = dXHat_rowMean.get(i, 0);
            double xh_dxh_rm = xHat_dXHat_rowMean.get(i, 0);
            
            for (int j = 0; j < D; j++) {
                double val = invStd * (dXHat.get(i, j) - dxh_rm - state.xHat.get(i, j) * xh_dxh_rm);
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
