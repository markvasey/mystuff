package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.CudaBridge;
import com.learnai.words.math.Matrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

public class GpuEmbeddingLayer implements GpuLayer {
    private GpuMatrix embeddings;
    private final GpuAdam opt;

    public GpuEmbeddingLayer(int vocabSize, int embeddingDim) {
        Matrix cpuEmb = Matrix.random(vocabSize, embeddingDim);
        this.embeddings = GpuMatrix.fromCpu(cpuEmb);
        this.opt = new GpuAdam(vocabSize, embeddingDim);
    }

    public GpuForwardResult forward(int[] tokenIds) {
        int numTokens = tokenIds.length;
        int dim = embeddings.getCols();
        GpuMatrix output = new GpuMatrix(numTokens, dim);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostIds = arena.allocateFrom(ValueLayout.JAVA_INT, tokenIds);
            long idsBytes = (long) numTokens * ValueLayout.JAVA_INT.byteSize();
            MemorySegment devIds = CudaBridge.cudaMalloc(idsBytes);
            try {
                CudaBridge.cudaMemcpyToDevice(devIds, hostIds, idsBytes);
                CudaBridge.cudaEmbeddingForward(embeddings.getDevicePtr(), devIds, output.getDevicePtr(), numTokens, dim);
            } finally {
                CudaBridge.cudaFree(devIds);
            }
        }
        return new GpuForwardResult(output, tokenIds);
    }

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        return null;
    }

    @Override
    public GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate) {
        int[] lastInputIds = (int[]) context;
        int vocabSize = embeddings.getRows();
        int dim = embeddings.getCols();
        int numTokens = lastInputIds.length;

        GpuMatrix gradient = new GpuMatrix(vocabSize, dim);
        float[] zeros = new float[vocabSize * dim];
        gradient.upload(zeros);

        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostIds = arena.allocateFrom(ValueLayout.JAVA_INT, lastInputIds);
            long idsBytes = (long) numTokens * ValueLayout.JAVA_INT.byteSize();
            MemorySegment devIds = CudaBridge.cudaMalloc(idsBytes);
            try {
                CudaBridge.cudaMemcpyToDevice(devIds, hostIds, idsBytes);
                CudaBridge.cudaEmbeddingBackward(outputGradient.getDevicePtr(), devIds, gradient.getDevicePtr(), numTokens, dim);
            } finally {
                CudaBridge.cudaFree(devIds);
            }
        }

        if (learningRate > 0) {
            opt.update(embeddings, gradient, learningRate);
        }

        gradient.close();
        return null; // Return null since embedding is the first layer
    }

    public int getEmbeddingDim() {
        return embeddings.getCols();
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        Matrix cpuEmb = embeddings.toCpu();
        cpuEmb.save(dos);
        opt.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        if (embeddings != null) embeddings.close();
        Matrix cpuEmb = Matrix.load(dis);
        this.embeddings = GpuMatrix.fromCpu(cpuEmb);
        opt.load(dis);
    }

    @Override
    public void close() {
        if (embeddings != null) embeddings.close();
        if (opt != null) opt.close();
    }
}
