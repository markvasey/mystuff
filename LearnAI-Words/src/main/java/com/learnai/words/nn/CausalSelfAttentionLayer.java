package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class CausalSelfAttentionLayer implements Layer {
    private Matrix wq, wk, wv;
    private final Adam qOpt, kOpt, vOpt;
    private final int d_model;

    private static class AttentionState {
        public final Matrix input;
        public final Matrix attnWeights;
        public AttentionState(Matrix input, Matrix attnWeights) {
            this.input = input;
            this.attnWeights = attnWeights;
        }
    }

    public CausalSelfAttentionLayer(int d_model) {
        this.d_model = d_model;
        this.wq = Matrix.random(d_model, d_model);
        this.wk = Matrix.random(d_model, d_model);
        this.wv = Matrix.random(d_model, d_model);
        this.qOpt = new Adam(d_model, d_model);
        this.kOpt = new Adam(d_model, d_model);
        this.vOpt = new Adam(d_model, d_model);
    }

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix q = input.multiply(wq);
        Matrix k = input.multiply(wk);
        Matrix v = input.multiply(wv);

        Matrix scores = q.multiply(k, false, true);
        double scale = Math.sqrt(d_model);
        int rows = scores.getRows();
        int cols = scores.getCols();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                scores.set(i, j, scores.get(i, j) / scale);
                if (j > i) scores.set(i, j, -1e9);
            }
        }

        Matrix attnWeights = new SoftmaxLayer().forward(scores).output;
        Matrix output = attnWeights.multiply(v);

        return new ForwardResult(output, new AttentionState(input, attnWeights));
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, double learningRate) {
        AttentionState state = (AttentionState) context;
        Matrix X = state.input;
        Matrix A = state.attnWeights;
        
        // 1. dL/dV = A^T * dOut
        Matrix dV = A.multiply(outputGradient, true, false); 
        Matrix dWv = X.multiply(dV, true, false);

        // 2. dL/dA = dOut * V^T
        Matrix V = X.multiply(wv);
        Matrix dA = outputGradient.multiply(V, false, true);
        
        // 3. dS (Softmax backward)
        Matrix dS = new Matrix(A.getRows(), A.getCols());
        for (int i = 0; i < A.getRows(); i++) {
            double dot = 0;
            for (int k_idx = 0; k_idx < A.getCols(); k_idx++) {
                dot += dA.get(i, k_idx) * A.get(i, k_idx);
            }
            for (int j = 0; j < A.getCols(); j++) {
                dS.set(i, j, A.get(i, j) * (dA.get(i, j) - dot));
            }
        }
        
        // 4. dQ, dK through Scaling and Masking
        double scale = 1.0 / Math.sqrt(d_model);
        for (int i = 0; i < dS.getRows(); i++) {
            for (int j = 0; j < dS.getCols(); j++) {
                dS.set(i, j, dS.get(i, j) * scale);
                if (j > i) dS.set(i, j, 0); 
            }
        }

        Matrix Q = X.multiply(wq);
        Matrix K = X.multiply(wk);
        
        Matrix dQ = dS.multiply(K);
        Matrix dK = dS.multiply(Q, true, false); // dS^T * Q
        
        Matrix dWq = X.multiply(dQ, true, false);
        Matrix dWk = X.multiply(dK, true, false);

        if (learningRate > 0) {
            qOpt.update(wq, dWq, learningRate);
            kOpt.update(wk, dWk, learningRate);
            vOpt.update(wv, dWv, learningRate);
        }

        // 5. Input Gradient (Backprop to X)
        // dX = dQ*Wq^T + dK*Wk^T + dV*Wv^T
        Matrix gradX = dQ.multiply(wq, false, true);
        gradX.addInPlace(dK.multiply(wk, false, true));
        gradX.addInPlace(dV.multiply(wv, false, true));

        return gradX;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        wq.save(dos); wk.save(dos); wv.save(dos);
        qOpt.save(dos); kOpt.save(dos); vOpt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        this.wq = Matrix.load(dis); this.wk = Matrix.load(dis); this.wv = Matrix.load(dis);
        this.qOpt.load(dis); this.kOpt.load(dis); this.vOpt.load(dis);
    }
}
