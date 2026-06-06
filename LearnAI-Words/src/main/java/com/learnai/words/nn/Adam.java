package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class Adam {
    private Matrix m; // 1st moment
    private Matrix v; // 2nd moment
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double eps = 1e-8;
    private int t = 0;

    public Adam(int r, int c) {
        this.m = new Matrix(r, c);
        this.v = new Matrix(r, c);
    }

    public synchronized void update(Matrix weights, Matrix gradient, double lr) {
        t++;
        double[] wData = weights.getData();
        double[] gData = gradient.getData();
        double[] mData = m.getData();
        double[] vData = v.getData();

        double biasCorr1 = 1.0 - Math.pow(beta1, t);
        double biasCorr2 = 1.0 - Math.pow(beta2, t);

        for (int i = 0; i < wData.length; i++) {
            mData[i] = beta1 * mData[i] + (1.0 - beta1) * gData[i];
            vData[i] = beta2 * vData[i] + (1.0 - beta2) * gData[i] * gData[i];
            
            double mHat = mData[i] / biasCorr1;
            double vHat = vData[i] / biasCorr2;
            
            wData[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    public void save(DataOutputStream dos) throws IOException { m.save(dos); v.save(dos); dos.writeInt(t); }
    public void load(DataInputStream dis) throws IOException { m = Matrix.load(dis); v = Matrix.load(dis); t = dis.readInt(); }
}
