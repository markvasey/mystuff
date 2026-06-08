package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Adam {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
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
        double[] w = weights.getData();
        double[] g = gradient.getData();
        double[] mData = m.getData();
        double[] vData = v.getData();

        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);

        double oneMinusBeta1 = 1.0 - beta1;
        double oneMinusBeta2 = 1.0 - beta2;

        int i = 0;
        int limit = SPECIES.loopBound(w.length);
        if (limit > 0) {
            var v_beta1 = DoubleVector.broadcast(SPECIES, beta1);
            var v_oneMinusBeta1 = DoubleVector.broadcast(SPECIES, oneMinusBeta1);
            var v_beta2 = DoubleVector.broadcast(SPECIES, beta2);
            var v_oneMinusBeta2 = DoubleVector.broadcast(SPECIES, oneMinusBeta2);
            var v_bc1 = DoubleVector.broadcast(SPECIES, bc1);
            var v_bc2 = DoubleVector.broadcast(SPECIES, bc2);
            var v_lr = DoubleVector.broadcast(SPECIES, lr);
            var v_eps = DoubleVector.broadcast(SPECIES, eps);

            for (; i < limit; i += SPECIES.length()) {
                var v_w = DoubleVector.fromArray(SPECIES, w, i);
                var v_g = DoubleVector.fromArray(SPECIES, g, i);
                var v_m = DoubleVector.fromArray(SPECIES, mData, i);
                var v_v = DoubleVector.fromArray(SPECIES, vData, i);

                // Update moments
                var v_m_new = v_m.mul(v_beta1).add(v_g.mul(v_oneMinusBeta1));
                var v_v_new = v_v.mul(v_beta2).add(v_g.mul(v_g).mul(v_oneMinusBeta2));

                v_m_new.intoArray(mData, i);
                v_v_new.intoArray(vData, i);

                // Bias corrections
                var v_mHat = v_m_new.div(v_bc1);
                var v_vHat = v_v_new.div(v_bc2);

                // Weight update
                var v_w_new = v_w.sub(v_lr.mul(v_mHat).div(v_vHat.sqrt().add(v_eps)));
                v_w_new.intoArray(w, i);
            }
        }

        // Tail loop for remaining elements
        for (; i < w.length; i++) {
            double grad = g[i];
            mData[i] = beta1 * mData[i] + oneMinusBeta1 * grad;
            vData[i] = beta2 * vData[i] + oneMinusBeta2 * grad * grad;
            
            double mHat = mData[i] / bc1;
            double vHat = vData[i] / bc2;
            
            w[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    public void save(DataOutputStream dos) throws IOException { m.save(dos); v.save(dos); dos.writeInt(t); }
    public void load(DataInputStream dis) throws IOException { m = Matrix.load(dis); v = Matrix.load(dis); t = dis.readInt(); }
}
