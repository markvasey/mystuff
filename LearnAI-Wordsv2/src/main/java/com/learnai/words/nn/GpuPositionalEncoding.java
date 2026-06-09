package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GpuPositionalEncoding implements GpuLayer {
    private GpuMatrix pe;

    public GpuPositionalEncoding(int maxLen, int d_model) {
        Matrix cpuPe = new Matrix(maxLen, d_model);
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < d_model; i++) {
                float val = (i % 2 == 0) ? 
                    (float) Math.sin(pos / Math.pow(10000, (double) i / d_model)) :
                    (float) Math.cos(pos / Math.pow(10000, (double) (i - 1) / d_model));
                cpuPe.set(pos, i, val);
            }
        }
        this.pe = GpuMatrix.fromCpu(cpuPe);
    }

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        // output = input + pe
        GpuMatrix output = input.add(pe);
        return new GpuForwardResult(output, null);
    }

    @Override
    public GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate) {
        GpuMatrix copy = new GpuMatrix(outputGradient.getRows(), outputGradient.getCols());
        long byteSize = (long) outputGradient.getRows() * outputGradient.getCols() * java.lang.foreign.ValueLayout.JAVA_FLOAT.byteSize();
        com.learnai.words.math.CudaBridge.cudaMemcpyToDevice(copy.getDevicePtr(), outputGradient.getDevicePtr(), byteSize);
        return copy;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        Matrix cpuPe = pe.toCpu();
        cpuPe.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        if (pe != null) pe.close();
        Matrix cpuPe = Matrix.load(dis);
        this.pe = GpuMatrix.fromCpu(cpuPe);
    }

    @Override
    public void close() {
        if (pe != null) {
            pe.close();
        }
    }
}
