package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

interface Layer {
    class ForwardResult {
        public final Matrix output;
        public final Object context;

        public ForwardResult(Matrix output, Object context) {
            this.output = output;
            this.context = context;
        }
    }

    ForwardResult forward(Matrix input);
    Matrix backward(Matrix outputGradient, Object context, float learningRate);
    void save(DataOutputStream dos) throws IOException;
    void load(DataInputStream dis) throws IOException;
}

class Adam {
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

class DenseLayer implements Layer {
    private Matrix weights;
    private Matrix bias;
    private final Adam weightsOpt;
    private final Adam biasOpt;

    public DenseLayer(int inputDim, int outputDim) {
        this.weights = Matrix.random(inputDim, outputDim);
        this.bias = new Matrix(1, outputDim);
        this.weightsOpt = new Adam(inputDim, outputDim);
        this.biasOpt = new Adam(1, outputDim);
    }

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix output = input.multiply(weights).add(bias);
        return new ForwardResult(output, input);
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        Matrix lastInput = (Matrix) context;
        Matrix weightsGradient = lastInput.multiply(outputGradient, true, false);
        Matrix biasGradient = new Matrix(1, bias.getCols());
        float[] bg = biasGradient.getData();
        float[] og = outputGradient.getData();
        int ogRows = outputGradient.getRows();
        int ogCols = outputGradient.getCols();
        
        for (int i = 0; i < ogRows; i++) {
            int off = i * ogCols;
            for (int j = 0; j < ogCols; j++) {
                bg[j] += og[off + j];
            }
        }

        Matrix inputGradient = outputGradient.multiply(weights, false, true);

        if (learningRate > 0) {
            weightsOpt.update(weights, weightsGradient, learningRate);
            biasOpt.update(bias, biasGradient, learningRate);
        }

        return inputGradient;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        weights.save(dos);
        bias.save(dos);
        weightsOpt.save(dos);
        biasOpt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        this.weights = Matrix.load(dis);
        this.bias = Matrix.load(dis);
        this.weightsOpt.load(dis);
        this.biasOpt.load(dis);
    }
}

class LayerNorm implements Layer {
    private Matrix gamma;
    private Matrix beta;
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
        for (int i = 0; i < dim; i++) this.gamma.set(0, i, 1.0f);
        this.beta = new Matrix(1, dim);
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

class SoftmaxLayer implements Layer {
    @Override
    public ForwardResult forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());
        for (int i = 0; i < input.getRows(); i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < input.getCols(); j++) {
                if (input.get(i, j) > max) max = input.get(i, j);
            }

            float sum = 0.0f;
            for (int j = 0; j < input.getCols(); j++) {
                float val = (float) Math.exp(input.get(i, j) - max);
                output.set(i, j, val);
                sum += val;
            }

            for (int j = 0; j < input.getCols(); j++) {
                float prob = output.get(i, j) / sum;
                output.set(i, j, Math.clamp(prob, 1e-15f, 1.0f - 1e-15f));
            }
        }
        return new ForwardResult(output, output);
    }

    @Override
    public Matrix backward(Matrix target, Object context, float learningRate) {
        Matrix lastOutput = (Matrix) context;
        Matrix gradient = new Matrix(lastOutput.getRows(), lastOutput.getCols());
        for (int i = 0; i < lastOutput.getRows(); i++) {
            for (int j = 0; j < lastOutput.getCols(); j++) {
                gradient.set(i, j, lastOutput.get(i, j) - target.get(i, j));
            }
        }
        return gradient;
    }

    @Override public void save(DataOutputStream dos) throws IOException {}
    @Override public void load(DataInputStream dis) throws IOException {}
}

class EmbeddingLayer implements Layer {
    private Matrix embeddings;
    private final Adam opt;

    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this.embeddings = Matrix.random(vocabSize, embeddingDim);
        this.opt = new Adam(vocabSize, embeddingDim);
    }

    public ForwardResult forward(int[] tokenIds) {
        Matrix output = new Matrix(tokenIds.length, embeddings.getCols());
        for (int i = 0; i < tokenIds.length; i++) {
            int id = tokenIds[i];
            for (int j = 0; j < embeddings.getCols(); j++) {
                output.set(i, j, embeddings.get(id, j));
            }
        }
        return new ForwardResult(output, tokenIds);
    }

    @Override public ForwardResult forward(Matrix input) { return null; }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        int[] lastInputIds = (int[]) context;
        int vocabSize = embeddings.getRows();
        int dim = embeddings.getCols();
        Matrix gradient = new Matrix(vocabSize, dim);
        for (int i = 0; i < lastInputIds.length; i++) {
            int id = lastInputIds[i];
            for (int j = 0; j < dim; j++) {
                gradient.set(id, j, gradient.get(id, j) + outputGradient.get(i, j));
            }
        }
        if (learningRate > 0) opt.update(embeddings, gradient, learningRate);
        return null; 
    }

    public int getEmbeddingDim() { return embeddings.getCols(); }

    @Override public void save(DataOutputStream dos) throws IOException { embeddings.save(dos); opt.save(dos); }
    @Override public void load(DataInputStream dis) throws IOException { this.embeddings = Matrix.load(dis); this.opt.load(dis); }
}

class CausalSelfAttentionLayer implements Layer {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
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

    private final int maxLen;

    public CausalSelfAttentionLayer(int d_model) {
        this(d_model, 64);
    }

    public CausalSelfAttentionLayer(int d_model, int maxLen) {
        this.d_model = d_model;
        this.maxLen = maxLen;
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

        int B = input.getRows() / maxLen;
        int T = maxLen;
        if (input.getRows() % maxLen != 0 || input.getRows() < maxLen) {
            B = 1;
            T = input.getRows();
        }

        Matrix scores = new Matrix(B * T, T);
        float[] sData = scores.getData();
        float[] qData = q.getData();
        float[] kData = k.getData();

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int qRow = offset + i;
                for (int j = 0; j < T; j++) {
                    int kRow = offset + j;
                    float sum = 0.0f;
                    for (int d = 0; d < d_model; d++) {
                        sum += qData[qRow * d_model + d] * kData[kRow * d_model + d];
                    }
                    sData[qRow * T + j] = sum;
                }
            }
        }

        float scale = (float) Math.sqrt(d_model);
        float invScale = 1.0f / scale;
        int rows = scores.getRows();
        int cols = scores.getCols();
        for (int i = 0; i < rows; i++) {
            int off = i * cols;
            int seq_pos = i % T;
            int activeLimit = seq_pos + 1;
            int j = 0;
            var vInvScale = FloatVector.broadcast(SPECIES, invScale);
            for (; j < SPECIES.loopBound(activeLimit); j += SPECIES.length()) {
                var vVec = FloatVector.fromArray(SPECIES, sData, off + j);
                vVec.mul(vInvScale).intoArray(sData, off + j);
            }
            for (; j < activeLimit; j++) {
                sData[off + j] *= invScale;
            }
            if (activeLimit < cols) {
                java.util.Arrays.fill(sData, off + activeLimit, off + cols, -1e9f);
            }
        }

        Matrix attnWeights = new SoftmaxLayer().forward(scores).output;

        Matrix output = new Matrix(B * T, d_model);
        float[] outData = output.getData();
        float[] wData = attnWeights.getData();
        float[] vData = v.getData();

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int outRow = offset + i;
                for (int d = 0; d < d_model; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < T; j++) {
                        sum += wData[outRow * T + j] * vData[(offset + j) * d_model + d];
                    }
                    outData[outRow * d_model + d] = sum;
                }
            }
        }

        return new ForwardResult(output, new AttentionState(input, attnWeights));
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        AttentionState state = (AttentionState) context;
        Matrix X = state.input;
        Matrix A = state.attnWeights;

        int B = X.getRows() / maxLen;
        int T = maxLen;
        if (X.getRows() % maxLen != 0 || X.getRows() < maxLen) {
            B = 1;
            T = X.getRows();
        }

        // 1. dL/dV = A^T * dOut (block-wise)
        Matrix dV = new Matrix(B * T, d_model);
        float[] dvData = dV.getData();
        float[] aData = A.getData();
        float[] ogData = outputGradient.getData();

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int dvRow = offset + i;
                for (int d = 0; d < d_model; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < T; j++) {
                        sum += aData[(offset + j) * T + i] * ogData[(offset + j) * d_model + d];
                    }
                    dvData[dvRow * d_model + d] = sum;
                }
            }
        }
        Matrix dWv = X.multiply(dV, true, false);

        // 2. dL/dA = dOut * V^T (block-wise)
        Matrix V = X.multiply(wv);
        Matrix dA = new Matrix(B * T, T);
        float[] daData = dA.getData();
        float[] vData = V.getData();

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int daRow = offset + i;
                for (int j = 0; j < T; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < d_model; d++) {
                        sum += ogData[daRow * d_model + d] * vData[(offset + j) * d_model + d];
                    }
                    daData[daRow * T + j] = sum;
                }
            }
        }

        // 3. dS (Softmax backward + Scaling)
        Matrix dS = new Matrix(B * T, T);
        float[] dsData = dS.getData();
        float scale = (float) (1.0 / Math.sqrt(d_model));

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int row = offset + i;
                float dot = 0.0f;
                for (int k = 0; k <= i; k++) {
                    dot += daData[row * T + k] * aData[row * T + k];
                }
                for (int j = 0; j <= i; j++) {
                    dsData[row * T + j] = aData[row * T + j] * (daData[row * T + j] - dot) * scale;
                }
            }
        }

        // 4. dQ, dK (block-wise)
        Matrix Q = X.multiply(wq);
        Matrix K = X.multiply(wk);

        Matrix dQ = new Matrix(B * T, d_model);
        float[] dqData = dQ.getData();
        float[] kData = K.getData();

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int dqRow = offset + i;
                for (int d = 0; d < d_model; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < T; j++) {
                        sum += dsData[dqRow * T + j] * kData[(offset + j) * d_model + d];
                    }
                    dqData[dqRow * d_model + d] = sum;
                }
            }
        }

        Matrix dK = new Matrix(B * T, d_model);
        float[] dkData = dK.getData();
        float[] qData = Q.getData();

        for (int b = 0; b < B; b++) {
            int offset = b * T;
            for (int i = 0; i < T; i++) {
                int dkRow = offset + i;
                for (int d = 0; d < d_model; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < T; j++) {
                        sum += dsData[(offset + j) * T + i] * qData[(offset + j) * d_model + d];
                    }
                    dkData[dkRow * d_model + d] = sum;
                }
            }
        }

        Matrix dWq = X.multiply(dQ, true, false);
        Matrix dWk = X.multiply(dK, true, false);

        if (learningRate > 0) {
            qOpt.update(wq, dWq, learningRate);
            kOpt.update(wk, dWk, learningRate);
            vOpt.update(wv, dWv, learningRate);
        }

        // 5. Input Gradient (Backprop to X)
        Matrix gradX = dQ.multiply(wq, false, true);
        gradX.addInPlace(dK.multiply(wk, false, true));
        gradX.addInPlace(dV.multiply(wv, false, true));

        return gradX;
    }

    @Override public void save(DataOutputStream dos) throws IOException { wq.save(dos); wk.save(dos); wv.save(dos); qOpt.save(dos); kOpt.save(dos); vOpt.save(dos); }
    @Override public void load(DataInputStream dis) throws IOException { this.wq = Matrix.load(dis); this.wk = Matrix.load(dis); this.wv = Matrix.load(dis); this.qOpt.load(dis); this.kOpt.load(dis); this.vOpt.load(dis); }
}

class PositionalEncoding implements Layer {
    private Matrix pe;

    public PositionalEncoding(int maxLen, int d_model) {
        this.pe = new Matrix(maxLen, d_model);
        for (int pos = 0; pos < maxLen; pos++) {
            for (int i = 0; i < d_model; i++) {
                float val = (i % 2 == 0) ? 
                    (float) Math.sin(pos / Math.pow(10000, (double) i / d_model)) :
                    (float) Math.cos(pos / Math.pow(10000, (double) (i - 1) / d_model));
                pe.set(pos, i, val);
            }
        }
    }

    @Override
    public ForwardResult forward(Matrix input) {
        Matrix output = new Matrix(input.getRows(), input.getCols());
        int maxLen = pe.getRows();
        for (int i = 0; i < input.getRows(); i++) {
            int pos = i % maxLen;
            for (int j = 0; j < input.getCols(); j++) {
                output.set(i, j, input.get(i, j) + pe.get(pos, j));
            }
        }
        return new ForwardResult(output, null);
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        return outputGradient;
    }

    @Override public void save(DataOutputStream dos) throws IOException { pe.save(dos); }
    @Override public void load(DataInputStream dis) throws IOException { this.pe = Matrix.load(dis); }
}

class ResidualBlock implements Layer {
    private final Layer sublayer;

    public ResidualBlock(Layer sublayer) {
        this.sublayer = sublayer;
    }

    @Override
    public ForwardResult forward(Matrix input) {
        ForwardResult res = sublayer.forward(input);
        Matrix output = res.output.add(input);
        return new ForwardResult(output, res.context);
    }

    @Override
    public Matrix backward(Matrix outputGradient, Object context, float learningRate) {
        Matrix sublayerGrad = sublayer.backward(outputGradient, context, learningRate);
        return sublayerGrad.add(outputGradient);
    }

    @Override public void save(DataOutputStream dos) throws IOException { sublayer.save(dos); }
    @Override public void load(DataInputStream dis) throws IOException { sublayer.load(dis); }
}
