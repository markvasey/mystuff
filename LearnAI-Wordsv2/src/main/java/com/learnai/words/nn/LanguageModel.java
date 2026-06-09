package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.util.ArrayList;
import java.util.List;

public class LanguageModel {
    private final EmbeddingLayer embedding;
    private final PositionalEncoding positional;
    private final List<Layer> layers = new ArrayList<>();
    private final SoftmaxLayer softmax = new SoftmaxLayer();
    private int completedEpochs = 0;

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
        
        // 3 Transformer Blocks for Phase 2
        for (int i = 0; i < 3; i++) {
            layers.add(new LayerNorm(d_model));
            layers.add(new ResidualBlock(new CausalSelfAttentionLayer(d_model)));
            layers.add(new LayerNorm(d_model));
            layers.add(new ResidualBlock(new DenseLayer(d_model, d_model)));
        }

        // Output Head
        layers.add(new LayerNorm(d_model));
        layers.add(new DenseLayer(d_model, vocabSize));
    }

    public int getCompletedEpochs() { return completedEpochs; }
    public void setCompletedEpochs(int n) { this.completedEpochs = n; }

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

    public float train(int[] tokenIds, int targetId, float learningRate) {
        ModelForwardResult fwd = forward(tokenIds);
        Matrix probs = fwd.finalProbs;
        int seqLen = tokenIds.length;
        int vocabSize = probs.getCols();

        // Calculate average loss over all sequence positions (predicting next token)
        float loss = 0.0f;
        Matrix target = new Matrix(seqLen, vocabSize);
        for (int i = 0; i < seqLen - 1; i++) {
            int nextTokenId = tokenIds[i + 1];
            float prob = Math.clamp(probs.get(i, nextTokenId), 1e-12f, 1.0f);
            loss += (float) -Math.log(prob);
            target.set(i, nextTokenId, 1.0f);
        }
        float finalProb = Math.clamp(probs.get(seqLen - 1, targetId), 1e-12f, 1.0f);
        loss += (float) -Math.log(finalProb);
        target.set(seqLen - 1, targetId, 1.0f);
        loss /= seqLen;

        Matrix gradient = softmax.backward(target, fwd.finalProbs, learningRate);

        // Scale gradient by 1/seqLen because the loss is averaged over all seqLen positions
        float scale = 1.0f / seqLen;
        float[] gData = gradient.getData();
        for (int i = 0; i < gData.length; i++) {
            gData[i] *= scale;
        }

        // Global Gradient Clipping (Clip norm to 1.0)
        float norm = (float) Math.sqrt(gradient.square().rowMean().rowMean().get(0, 0));
        if (norm > 1.0f) {
            float clipScale = 1.0f / norm;
            float[] d = gradient.getData();
            for (int i = 0; i < d.length; i++) d[i] *= clipScale;
        }
        
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient, fwd.layerContexts.get(i), learningRate);
        }
        positional.backward(gradient, fwd.positionalContext, learningRate);
        embedding.backward(gradient, fwd.embeddingContext, learningRate);

        return loss;
    }
    
    public void save(String path) throws java.io.IOException {
        try (java.io.DataOutputStream dos = new java.io.DataOutputStream(new java.io.FileOutputStream(path))) {
            dos.writeInt(completedEpochs);
            embedding.save(dos);
            positional.save(dos);
            dos.writeInt(layers.size());
            for (Layer layer : layers) layer.save(dos);
        }
    }

    public void load(String path) throws java.io.IOException {
        try (java.io.DataInputStream dis = new java.io.DataInputStream(new java.io.FileInputStream(path))) {
            this.completedEpochs = dis.readInt();
            embedding.load(dis);
            positional.load(dis);
            int layerCount = dis.readInt();
            for (int i = 0; i < layerCount; i++) layers.get(i).load(dis);
        }
    }
}
