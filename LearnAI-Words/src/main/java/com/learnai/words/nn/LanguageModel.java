package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.util.ArrayList;
import java.util.List;

public class LanguageModel {
    private final EmbeddingLayer embedding;
    private final PositionalEncoding positional;
    private final List<Layer> layers = new ArrayList<>();
    private final SoftmaxLayer softmax = new SoftmaxLayer();

    private static class ModelForwardResult {
        public final Matrix finalProbs;
        public final List<Object> layerContexts;
        public final Object embeddingContext;
        public final Object positionalContext;

        public ModelForwardResult(Matrix finalProbs, List<Object> layerContexts, Object embeddingContext, Object positionalContext) {
            this.finalProbs = finalProbs;
            this.layerContexts = layerContexts;
            this.embeddingContext = embeddingContext;
            this.positionalContext = positionalContext;
        }
    }

    public LanguageModel(int vocabSize, int d_model, int maxLen) {
        this.embedding = new EmbeddingLayer(vocabSize, d_model);
        this.positional = new PositionalEncoding(maxLen, d_model);
        
        // Block 1: Attention + FeedForward
        layers.add(new LayerNorm(d_model));
        layers.add(new ResidualBlock(new CausalSelfAttentionLayer(d_model)));
        layers.add(new LayerNorm(d_model));
        layers.add(new ResidualBlock(new DenseLayer(d_model, d_model)));

        // Block 2: Attention + FeedForward
        layers.add(new LayerNorm(d_model));
        layers.add(new ResidualBlock(new CausalSelfAttentionLayer(d_model)));
        layers.add(new LayerNorm(d_model));
        layers.add(new ResidualBlock(new DenseLayer(d_model, d_model)));

        // Output Head
        layers.add(new LayerNorm(d_model));
        layers.add(new DenseLayer(d_model, vocabSize));
    }

    public Matrix predict(int[] tokenIds) {
        return forward(tokenIds).finalProbs;
    }

    private ModelForwardResult forward(int[] tokenIds) {
        Layer.ForwardResult embRes = embedding.forward(tokenIds);
        Layer.ForwardResult posRes = positional.forward(embRes.output);
        
        List<Object> contexts = new ArrayList<>();
        Matrix x = posRes.output;
        for (Layer layer : layers) {
            Layer.ForwardResult res = layer.forward(x);
            x = res.output;
            contexts.add(res.context);
        }
        
        Layer.ForwardResult softRes = softmax.forward(x);
        return new ModelForwardResult(softRes.output, contexts, embRes.context, posRes.context);
    }

    public double train(int[] tokenIds, int targetId, double learningRate) {
        // Forward (Thread-safe)
        ModelForwardResult fwd = forward(tokenIds);
        Matrix probs = fwd.finalProbs;
        
        int seqLen = tokenIds.length;
        Matrix target = new Matrix(seqLen, probs.getCols());
        target.set(seqLen - 1, targetId, 1.0);

        double loss = -Math.log(Math.max(probs.get(seqLen - 1, targetId), 1e-10));

        // Backward (Concurrent parts are thread-safe, weight updates are synchronized)
        Matrix fullGradient = softmax.backward(target, fwd.finalProbs, learningRate);
        
        Matrix lastTokenGradient = new Matrix(seqLen, fullGradient.getCols());
        for (int j = 0; j < fullGradient.getCols(); j++) {
            lastTokenGradient.set(seqLen - 1, j, fullGradient.get(seqLen - 1, j));
        }

        Matrix gradient = lastTokenGradient;
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient, fwd.layerContexts.get(i), learningRate);
        }
        positional.backward(gradient, fwd.positionalContext, learningRate);
        embedding.backward(gradient, fwd.embeddingContext, learningRate);

        return loss;
    }
    
    public EmbeddingLayer getEmbedding() { return embedding; }

    public void save(String path) throws java.io.IOException {
        try (java.io.DataOutputStream dos = new java.io.DataOutputStream(new java.io.FileOutputStream(path))) {
            embedding.save(dos);
            positional.save(dos);
            dos.writeInt(layers.size());
            for (Layer layer : layers) {
                layer.save(dos);
            }
        }
    }

    public void load(String path) throws java.io.IOException {
        try (java.io.DataInputStream dis = new java.io.DataInputStream(new java.io.FileInputStream(path))) {
            embedding.load(dis);
            positional.load(dis);
            int layerCount = dis.readInt();
            for (int i = 0; i < layerCount; i++) {
                layers.get(i).load(dis);
            }
        }
    }
}
