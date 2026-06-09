package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public interface GpuLayer extends AutoCloseable {
    class GpuForwardResult {
        public final GpuMatrix output;
        public final Object context;

        public GpuForwardResult(GpuMatrix output, Object context) {
            this.output = output;
            this.context = context;
        }
    }

    GpuForwardResult forward(GpuMatrix input);
    
    GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate);
    
    void save(DataOutputStream dos) throws IOException;
    void load(DataInputStream dis) throws IOException;
}
