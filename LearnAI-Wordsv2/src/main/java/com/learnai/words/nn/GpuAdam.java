package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.CudaBridge;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import com.learnai.words.math.Matrix;

public class GpuAdam implements AutoCloseable {
    private GpuMatrix m; // 1st moment
    private GpuMatrix v; // 2nd moment
    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float eps = 1e-8f;
    private int t = 0;

    public GpuAdam(int r, int c) {
        this.m = new GpuMatrix(r, c);
        this.v = new GpuMatrix(r, c);
        // Initialize moments to zero in GPU VRAM
        float[] zeros = new float[r * c];
        m.upload(zeros);
        v.upload(zeros);
    }

    public synchronized void update(GpuMatrix weights, GpuMatrix gradient, float lr) {
        t++;
        int size = weights.getRows() * weights.getCols();
        CudaBridge.cudaAdamUpdate(
            weights.getDevicePtr(),
            gradient.getDevicePtr(),
            m.getDevicePtr(),
            v.getDevicePtr(),
            size,
            lr,
            beta1,
            beta2,
            eps,
            t
        );
    }

    public void save(DataOutputStream dos) throws IOException {
        Matrix cpuM = m.toCpu();
        Matrix cpuV = v.toCpu();
        cpuM.save(dos);
        cpuV.save(dos);
        dos.writeInt(t);
    }

    public void load(DataInputStream dis) throws IOException {
        if (m != null) m.close();
        if (v != null) v.close();
        Matrix cpuM = Matrix.load(dis);
        Matrix cpuV = Matrix.load(dis);
        m = GpuMatrix.fromCpu(cpuM);
        v = GpuMatrix.fromCpu(cpuV);
        t = dis.readInt();
    }

    @Override
    public void close() {
        if (m != null) {
            m.close();
        }
        if (v != null) {
            v.close();
        }
    }
}
