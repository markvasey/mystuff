package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.CudaBridge;
import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GpuLayerNorm implements GpuLayer {
    private GpuMatrix gamma; // Gain
    private GpuMatrix beta;  // Bias
    private final GpuAdam gammaOpt;
    private final GpuAdam betaOpt;
    private final float eps = 1e-5f;

    private static class LNState implements AutoCloseable {
        public GpuMatrix input;
        public GpuMatrix xHat;
        public GpuMatrix var;
        public LNState(GpuMatrix i, GpuMatrix x, GpuMatrix v) {
            this.input = i;
            this.xHat = x;
            this.var = v;
        }
        @Override
        public void close() {
            if (input != null) { input.close(); input = null; }
            if (xHat != null) { xHat.close(); xHat = null; }
            if (var != null) { var.close(); var = null; }
        }
    }

    public GpuLayerNorm(int dim) {
        this.gamma = new GpuMatrix(1, dim);
        float[] initialGamma = new float[dim];
        for (int i = 0; i < dim; i++) initialGamma[i] = 1.0f;
        this.gamma.upload(initialGamma);

        this.beta = new GpuMatrix(1, dim);
        float[] initialBeta = new float[dim];
        this.beta.upload(initialBeta);

        this.gammaOpt = new GpuAdam(1, dim);
        this.betaOpt = new GpuAdam(1, dim);
    }

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        int N = input.getRows();
        int D = input.getCols();

        GpuMatrix mean = input.rowMean();
        GpuMatrix var = input.rowVariance(mean);

        GpuMatrix xHat = new GpuMatrix(N, D);
        GpuMatrix output = new GpuMatrix(N, D);

        CudaBridge.cudaLayerNormForward(
            input.getDevicePtr(),
            gamma.getDevicePtr(),
            beta.getDevicePtr(),
            output.getDevicePtr(),
            xHat.getDevicePtr(),
            mean.getDevicePtr(),
            var.getDevicePtr(),
            N,
            D,
            eps
        );

        GpuMatrix inputCopy = new GpuMatrix(N, D);
        long byteSize = (long) N * D * java.lang.foreign.ValueLayout.JAVA_FLOAT.byteSize();
        CudaBridge.cudaMemcpyToDevice(inputCopy.getDevicePtr(), input.getDevicePtr(), byteSize);

        mean.close(); // mean is no longer needed after forward pass

        return new GpuForwardResult(output, new LNState(inputCopy, xHat, var));
    }

    @Override
    public GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate) {
        LNState state = (LNState) context;
        int N = outputGradient.getRows();
        int D = outputGradient.getCols();

        GpuMatrix dGamma = new GpuMatrix(1, D);
        GpuMatrix dBeta = new GpuMatrix(1, D);
        GpuMatrix dInput = new GpuMatrix(N, D);

        CudaBridge.cudaLayerNormBackward(
            outputGradient.getDevicePtr(),
            state.xHat.getDevicePtr(),
            state.var.getDevicePtr(),
            gamma.getDevicePtr(),
            dInput.getDevicePtr(),
            dGamma.getDevicePtr(),
            dBeta.getDevicePtr(),
            N,
            D,
            eps
        );

        if (learningRate > 0) {
            gammaOpt.update(gamma, dGamma, learningRate);
            betaOpt.update(beta, dBeta, learningRate);
        }

        dGamma.close();
        dBeta.close();
        state.close(); // Free VRAM allocations held in state

        return dInput;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        Matrix cpuGamma = gamma.toCpu();
        Matrix cpuBeta = beta.toCpu();
        cpuGamma.save(dos);
        cpuBeta.save(dos);
        gammaOpt.save(dos);
        betaOpt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        if (gamma != null) gamma.close();
        if (beta != null) beta.close();

        Matrix cpuGamma = Matrix.load(dis);
        Matrix cpuBeta = Matrix.load(dis);
        this.gamma = GpuMatrix.fromCpu(cpuGamma);
        this.beta = GpuMatrix.fromCpu(cpuBeta);

        gammaOpt.load(dis);
        betaOpt.load(dis);
    }

    @Override
    public void close() {
        if (gamma != null) gamma.close();
        if (beta != null) beta.close();
        if (gammaOpt != null) gammaOpt.close();
        if (betaOpt != null) betaOpt.close();
    }
}
