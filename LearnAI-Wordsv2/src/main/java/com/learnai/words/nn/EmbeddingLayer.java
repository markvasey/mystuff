package com.learnai.words.nn;

import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class EmbeddingLayer implements Layer {
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
