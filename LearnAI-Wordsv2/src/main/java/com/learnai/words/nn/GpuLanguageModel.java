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

    private static class ModelForwardResult implements AutoCloseable {
        public final GpuMatrix finalProbs;
        public final List<Object> layerContexts;
        public final Object embeddingContext;
        public final Object positionalContext;
        public final Object softmaxContext;
        public final GpuMatrix[] activations;

        public ModelForwardResult(GpuMatrix finalProbs, List<Object> layerContexts, Object embeddingContext, Object positionalContext, Object softmaxContext, GpuMatrix[] activations) {
            this.finalProbs = finalProbs;
            this.layerContexts = layerContexts;
            this.embeddingContext = embeddingContext;
            this.positionalContext = positionalContext;
            this.softmaxContext = softmaxContext;
            this.activations = activations;
        }

        @Override
        public void close() {
            if (finalProbs != null) {
                finalProbs.close();
            }
            if (activations != null) {
                for (int i = 0; i < activations.length; i++) {
                    if (activations[i] != null) {
                        activations[i].close();
                        activations[i] = null;
                    }
                }
            }
            if (layerContexts != null) {
                for (Object ctx : layerContexts) {
                    if (ctx instanceof AutoCloseable) {
                        try {
                            ((AutoCloseable) ctx).close();
                        } catch (Exception e) {
                            // ignore
                        }
                    }
                }
                layerContexts.clear();
            }
            if (softmaxContext instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) softmaxContext).close();
                } catch (Exception e) {
                    // ignore
                }
            }
        }
    }

    public GpuLanguageModel(int vocabSize, int d_model, int maxLen) {
        this.embedding = new GpuEmbeddingLayer(vocabSize, d_model);
        this.positional = new GpuPositionalEncoding(maxLen, d_model);
        
        // 4 Transformer Blocks
        for (int i = 0; i < 4; i++) {
            layers.add(new GpuLayerNorm(d_model));
            layers.add(new GpuResidualBlock(new GpuCausalSelfAttentionLayer(d_model, maxLen)));
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
        try (ModelForwardResult fwd = forward(tokenIds)) {
            return fwd.finalProbs.toCpu();
        }
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
        activations[layers.size()] = null;
        
        return new ModelForwardResult(softRes.output, contexts, embRes.context, posRes.context, softRes.context, activations);
    }

    public float train(int[] tokenIds, int targetId, float learningRate) {
        return train(tokenIds, new int[]{targetId}, learningRate);
    }

    public float train(int[] tokenIds, int[] targetIds, float learningRate) {
        try (ModelForwardResult fwd = forward(tokenIds)) {
            GpuMatrix probs = fwd.finalProbs;
            int seqLen = tokenIds.length;
            int vocabSize = probs.getCols();

            int B = targetIds.length;
            int T = seqLen / B;

            // Construct target token index array for GPU side loss and backward calculation
            int[] targets = new int[seqLen];
            for (int b = 0; b < B; b++) {
                int offset = b * T;
                for (int i = 0; i < T - 1; i++) {
                    targets[offset + i] = tokenIds[offset + i + 1];
                }
                targets[offset + T - 1] = targetIds[b];
            }

            // Allocate gradient directly on device
            GpuMatrix gradient = new GpuMatrix(seqLen, vocabSize);

            // Perform softmax backward, loss computation, and global norm clipping entirely on the GPU
            float loss = com.learnai.words.math.CudaBridge.cudaSoftmaxBackwardLossClip(probs, targets, gradient);

            GpuMatrix currentGradient = gradient;
            for (int i = layers.size() - 1; i >= 0; i--) {
                GpuMatrix nextGrad = layers.get(i).backward(currentGradient, fwd.layerContexts.get(i), learningRate);
                currentGradient.close();
                currentGradient = nextGrad;
                
                if (fwd.activations[i] != null) {
                    fwd.activations[i].close();
                    fwd.activations[i] = null;
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
    }

    /**
     * Forward pass only — no backward pass, no weight updates.
     * Used for computing validation loss without affecting model parameters.
     */
    public float evaluate(int[] tokenIds, int[] targetIds) {
        try (ModelForwardResult fwd = forward(tokenIds)) {
            GpuMatrix probs = fwd.finalProbs;
            int seqLen = tokenIds.length;
            int vocabSize = probs.getCols();

            int B = targetIds.length;
            int T = seqLen / B;

            int[] targets = new int[seqLen];
            for (int b = 0; b < B; b++) {
                int offset = b * T;
                for (int i = 0; i < T - 1; i++) {
                    targets[offset + i] = tokenIds[offset + i + 1];
                }
                targets[offset + T - 1] = targetIds[b];
            }

            // Compute loss using a temporary gradient buffer (discarded immediately — no backward)
            GpuMatrix gradient = new GpuMatrix(seqLen, vocabSize);
            float loss = com.learnai.words.math.CudaBridge.cudaSoftmaxBackwardLossClip(probs, targets, gradient);
            gradient.close();
            return loss;
        }
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
