package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.CudaBridge;
import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GpuCausalSelfAttentionLayer implements GpuLayer {
    private GpuMatrix wq, wk, wv;
    private final GpuAdam qOpt, kOpt, vOpt;
    private final int d_model;

    private static class AttentionState implements AutoCloseable {
        public final GpuMatrix input;
        public final GpuMatrix attnWeights;
        public AttentionState(GpuMatrix input, GpuMatrix attnWeights) {
            this.input = input;
            this.attnWeights = attnWeights;
        }
        @Override
        public void close() {
            if (input != null) input.close();
            if (attnWeights != null) attnWeights.close();
        }
    }

    public GpuCausalSelfAttentionLayer(int d_model) {
        this.d_model = d_model;
        Matrix cpuWq = Matrix.random(d_model, d_model);
        Matrix cpuWk = Matrix.random(d_model, d_model);
        Matrix cpuWv = Matrix.random(d_model, d_model);

        this.wq = GpuMatrix.fromCpu(cpuWq);
        this.wk = GpuMatrix.fromCpu(cpuWk);
        this.wv = GpuMatrix.fromCpu(cpuWv);

        this.qOpt = new GpuAdam(d_model, d_model);
        this.kOpt = new GpuAdam(d_model, d_model);
        this.vOpt = new GpuAdam(d_model, d_model);
    }

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        GpuMatrix q = input.multiply(wq);
        GpuMatrix k = input.multiply(wk);
        GpuMatrix v = input.multiply(wv);

        GpuMatrix scores = q.multiply(k, false, true);
        q.close();
        k.close();

        float scale = (float) Math.sqrt(d_model);
        float invScale = 1.0f / scale;

        CudaBridge.cudaAttentionForward(scores.getDevicePtr(), scores.getRows(), scores.getCols(), invScale);

        GpuMatrix output = scores.multiply(v);
        v.close();

        GpuMatrix inputCopy = new GpuMatrix(input.getRows(), input.getCols());
        long byteSize = (long) input.getRows() * input.getCols() * java.lang.foreign.ValueLayout.JAVA_FLOAT.byteSize();
        CudaBridge.cudaMemcpyToDevice(inputCopy.getDevicePtr(), input.getDevicePtr(), byteSize);

        return new GpuForwardResult(output, new AttentionState(inputCopy, scores));
    }

    @Override
    public GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate) {
        AttentionState state = (AttentionState) context;
        GpuMatrix X = state.input;
        GpuMatrix A = state.attnWeights;

        // 1. dL/dV = A^T * dOut
        GpuMatrix dV = A.multiply(outputGradient, true, false);
        GpuMatrix dWv = X.multiply(dV, true, false);

        // 2. dL/dA = dOut * V^T
        GpuMatrix V = X.multiply(wv);
        GpuMatrix dA = outputGradient.multiply(V, false, true);
        V.close();

        // 3. dS (Softmax backward + Scaling + Masking)
        GpuMatrix dS = new GpuMatrix(A.getRows(), A.getCols());
        float scale = (float) (1.0 / Math.sqrt(d_model));
        CudaBridge.cudaAttentionBackward(A.getDevicePtr(), dA.getDevicePtr(), dS.getDevicePtr(), A.getRows(), A.getCols(), scale);
        dA.close();

        // 4. dQ, dK
        GpuMatrix Q = X.multiply(wq);
        GpuMatrix K = X.multiply(wk);

        GpuMatrix dQ = dS.multiply(K);
        GpuMatrix dK = dS.multiply(Q, true, false);
        Q.close();
        K.close();

        GpuMatrix dWq = X.multiply(dQ, true, false);
        GpuMatrix dWk = X.multiply(dK, true, false);

        if (learningRate > 0) {
            qOpt.update(wq, dWq, learningRate);
            kOpt.update(wk, dWk, learningRate);
            vOpt.update(wv, dWv, learningRate);
        }

        dWq.close();
        dWk.close();
        dWv.close();

        // 5. Input Gradient (Backprop to X)
        GpuMatrix gradX = dQ.multiply(wq, false, true);
        dQ.close();

        GpuMatrix dK_Wk = dK.multiply(wk, false, true);
        dK.close();
        gradX.addInPlace(dK_Wk);
        dK_Wk.close();

        GpuMatrix dV_Wv = dV.multiply(wv, false, true);
        dV.close();
        gradX.addInPlace(dV_Wv);
        dV_Wv.close();

        dS.close();
        state.close();

        return gradX;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        Matrix cpuWq = wq.toCpu();
        Matrix cpuWk = wk.toCpu();
        Matrix cpuWv = wv.toCpu();
        cpuWq.save(dos);
        cpuWk.save(dos);
        cpuWv.save(dos);

        qOpt.save(dos);
        kOpt.save(dos);
        vOpt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        if (wq != null) wq.close();
        if (wk != null) wk.close();
        if (wv != null) wv.close();

        Matrix cpuWq = Matrix.load(dis);
        Matrix cpuWk = Matrix.load(dis);
        Matrix cpuWv = Matrix.load(dis);

        this.wq = GpuMatrix.fromCpu(cpuWq);
        this.wk = GpuMatrix.fromCpu(cpuWk);
        this.wv = GpuMatrix.fromCpu(cpuWv);

        qOpt.load(dis);
        kOpt.load(dis);
        vOpt.load(dis);
    }

    @Override
    public void close() {
        if (wq != null) wq.close();
        if (wk != null) wk.close();
        if (wv != null) wv.close();
        if (qOpt != null) qOpt.close();
        if (kOpt != null) kOpt.close();
        if (vOpt != null) vOpt.close();
    }
}
