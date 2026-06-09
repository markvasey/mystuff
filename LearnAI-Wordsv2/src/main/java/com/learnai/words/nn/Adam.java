package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public class Adam {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private Matrix m; // 1st moment
    private Matrix v; // 2nd moment
    private final float beta1 = 0.9f;
    private final float beta2 = 0.999f;
    private final float eps = 1e-8f;
    private int t = 0;

    public Adam(int r, int c) {
        this.m = new Matrix(r, c);
        this.v = new Matrix(r, c);
    }

    public synchronized void update(Matrix weights, Matrix gradient, float lr) {
        t++;
        float[] w = weights.getData();
        float[] g = gradient.getData();
        float[] mData = m.getData();
        float[] vData = v.getData();

        float bc1 = (float) (1.0 - Math.pow(beta1, t));
        float bc2 = (float) (1.0 - Math.pow(beta2, t));

        float oneMinusBeta1 = 1.0f - beta1;
        float oneMinusBeta2 = 1.0f - beta2;

        int i = 0;
        int limit = SPECIES.loopBound(w.length);
        if (limit > 0) {
            var v_beta1 = FloatVector.broadcast(SPECIES, beta1);
            var v_oneMinusBeta1 = FloatVector.broadcast(SPECIES, oneMinusBeta1);
            var v_beta2 = FloatVector.broadcast(SPECIES, beta2);
            var v_oneMinusBeta2 = FloatVector.broadcast(SPECIES, oneMinusBeta2);
            var v_bc1 = FloatVector.broadcast(SPECIES, bc1);
            var v_bc2 = FloatVector.broadcast(SPECIES, bc2);
            var v_lr = FloatVector.broadcast(SPECIES, lr);
            var v_eps = FloatVector.broadcast(SPECIES, eps);

            for (; i < limit; i += SPECIES.length()) {
                var v_w = FloatVector.fromArray(SPECIES, w, i);
                var v_g = FloatVector.fromArray(SPECIES, g, i);
                var v_m = FloatVector.fromArray(SPECIES, mData, i);
                var v_v = FloatVector.fromArray(SPECIES, vData, i);

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
            float grad = g[i];
            mData[i] = beta1 * mData[i] + oneMinusBeta1 * grad;
            vData[i] = beta2 * vData[i] + oneMinusBeta2 * grad * grad;
            
            float mHat = mData[i] / bc1;
            float vHat = vData[i] / bc2;
            
            w[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
        }
    }

    public void save(DataOutputStream dos) throws IOException { m.save(dos); v.save(dos); dos.writeInt(t); }
    public void load(DataInputStream dis) throws IOException { m = Matrix.load(dis); v = Matrix.load(dis); t = dis.readInt(); }
}
