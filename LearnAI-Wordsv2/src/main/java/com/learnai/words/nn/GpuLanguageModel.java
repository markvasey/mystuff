package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.Matrix;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class GpuLanguageModel implements AutoCloseable {
    private final GpuEmbeddingLayer embedding;
    private final GpuPositionalEncoding positional;
    private final List<GpuLayer> layers = new ArrayList<>();
    private final GpuSoftmaxLayer softmax = new GpuSoftmaxLayer();
    private int completedEpochs = 0;

    private static class ModelForwardResult {
        public final GpuMatrix finalProbs;
        public final List<Object> layerContexts;
        public final Object embeddingContext;
        public final Object positionalContext;
        public final GpuMatrix[] activations;

        public ModelForwardResult(GpuMatrix finalProbs, List<Object> layerContexts, Object embeddingContext, Object positionalContext, GpuMatrix[] activations) {
            this.finalProbs = finalProbs;
            this.layerContexts = layerContexts;
            this.embeddingContext = embeddingContext;
            this.positionalContext = positionalContext;
            this.activations = activations;
        }
    }

    public GpuLanguageModel(int vocabSize, int d_model, int maxLen) {
        this.embedding = new GpuEmbeddingLayer(vocabSize, d_model);
        this.positional = new GpuPositionalEncoding(maxLen, d_model);
        
        // 3 Transformer Blocks
        for (int i = 0; i < 3; i++) {
            layers.add(new GpuLayerNorm(d_model));
            layers.add(new GpuResidualBlock(new GpuCausalSelfAttentionLayer(d_model)));
            layers.add(new GpuLayerNorm(d_model));
            layers.add(new GpuResidualBlock(new GpuDenseLayer(d_model, d_model)));
        }

        // Output Head
        layers.add(new GpuLayerNorm(d_model));
        layers.add(new GpuDenseLayer(d_model, vocabSize));
    }

    public int getCompletedEpochs() { return completedEpochs; }
    public void setCompletedEpochs(int n) { this.completedEpochs = n; }

    public Matrix predict(int[] tokenIds) {
        ModelForwardResult fwd = forward(tokenIds);
        Matrix cpuProbs = fwd.finalProbs.toCpu();
        fwd.finalProbs.close();
        return cpuProbs;
    }

    private ModelForwardResult forward(int[] tokenIds) {
        GpuLayer.GpuForwardResult embRes = embedding.forward(tokenIds);
        GpuMatrix embOut = embRes.output;
        
        GpuLayer.GpuForwardResult posRes = positional.forward(embOut);
        GpuMatrix posOut = posRes.output;
        embOut.close();
        
        GpuMatrix[] activations = new GpuMatrix[layers.size() + 1];
        activations[0] = posOut;
        
        List<Object> contexts = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            GpuLayer.GpuForwardResult res = layers.get(i).forward(activations[i]);
            activations[i + 1] = res.output;
            contexts.add(res.context);
        }
        
        GpuLayer.GpuForwardResult softRes = softmax.forward(activations[layers.size()]);
        activations[layers.size()].close();
        
        return new ModelForwardResult(softRes.output, contexts, embRes.context, posRes.context, activations);
    }

    public float train(int[] tokenIds, int targetId, float learningRate) {
        ModelForwardResult fwd = forward(tokenIds);
        GpuMatrix probs = fwd.finalProbs;
        int seqLen = tokenIds.length;
        int vocabSize = probs.getCols();

        // Download probs to calculate loss and build the target distribution
        float[] probsData = new float[seqLen * vocabSize];
        probs.download(probsData);

        float loss = 0.0f;
        Matrix cpuTarget = new Matrix(seqLen, vocabSize);
        for (int i = 0; i < seqLen - 1; i++) {
            int nextTokenId = tokenIds[i + 1];
            float prob = Math.clamp(probsData[i * vocabSize + nextTokenId], 1e-12f, 1.0f);
            loss += (float) -Math.log(prob);
            cpuTarget.set(i, nextTokenId, 1.0f);
        }
        float finalProb = Math.clamp(probsData[(seqLen - 1) * vocabSize + targetId], 1e-12f, 1.0f);
        loss += (float) -Math.log(finalProb);
        cpuTarget.set(seqLen - 1, targetId, 1.0f);
        loss /= seqLen;

        GpuMatrix target = GpuMatrix.fromCpu(cpuTarget);
        GpuMatrix gradient = softmax.backward(target, fwd.finalProbs, learningRate);
        target.close();

        // Download gradient data to perform scaling and global norm clipping on CPU
        float[] gData = new float[seqLen * vocabSize];
        gradient.download(gData);

        float scale = 1.0f / seqLen;
        for (int i = 0; i < gData.length; i++) {
            gData[i] *= scale;
        }

        // Global Gradient Clipping norm (RMS of first sequence position's predictions)
        float sumSq = 0.0f;
        for (int j = 0; j < vocabSize; j++) {
            float val = gData[j];
            sumSq += val * val;
        }
        float norm = (float) Math.sqrt(sumSq / vocabSize);

        if (norm > 1.0f) {
            float clipScale = 1.0f / norm;
            for (int i = 0; i < gData.length; i++) {
                gData[i] *= clipScale;
            }
        }

        gradient.upload(gData);

        GpuMatrix currentGradient = gradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            GpuMatrix nextGrad = layers.get(i).backward(currentGradient, fwd.layerContexts.get(i), learningRate);
            currentGradient.close();
            currentGradient = nextGrad;
            
            if (fwd.activations[i] != null) {
                fwd.activations[i].close();
            }
        }
        
        GpuMatrix posGrad = positional.backward(currentGradient, fwd.positionalContext, learningRate);
        currentGradient.close();
        currentGradient = posGrad;
        
        GpuMatrix embGrad = embedding.backward(currentGradient, fwd.embeddingContext, learningRate);
        currentGradient.close();
        if (embGrad != null) {
            embGrad.close();
        }

        return loss;
    }

    public void save(String path) throws IOException {
        try (java.io.DataOutputStream dos = new java.io.DataOutputStream(new java.io.FileOutputStream(path))) {
            dos.writeInt(completedEpochs);
            embedding.save(dos);
            positional.save(dos);
            dos.writeInt(layers.size());
            for (GpuLayer layer : layers) {
                layer.save(dos);
            }
        }
    }

    public void load(String path) throws IOException {
        try (java.io.DataInputStream dis = new java.io.DataInputStream(new java.io.FileInputStream(path))) {
            this.completedEpochs = dis.readInt();
            embedding.load(dis);
            positional.load(dis);
            int layerCount = dis.readInt();
            for (int i = 0; i < layerCount; i++) {
                layers.get(i).load(dis);
            }
        }
    }

    @Override
    public void close() {
        if (embedding != null) {
            embedding.close();
        }
        if (positional != null) {
            positional.close();
        }
        for (GpuLayer layer : layers) {
            if (layer != null) {
                try {
                    layer.close();
                } catch (Exception e) {
                    // Ignore
                }
            }
        }
        if (softmax != null) {
            try {
                softmax.close();
            } catch (Exception e) {
                // Ignore
            }
        }
    }
}
