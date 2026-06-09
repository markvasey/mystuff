package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GpuResidualBlock implements GpuLayer {
    private final GpuLayer sublayer;

    public GpuResidualBlock(GpuLayer sublayer) {
        this.sublayer = sublayer;
    }

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        GpuForwardResult res = sublayer.forward(input);
        GpuMatrix sublayerOut = res.output;
        GpuMatrix output = sublayerOut.add(input); // Allocates a new GpuMatrix
        sublayerOut.close(); // Free the intermediate matrix from sublayer forward
        return new GpuForwardResult(output, res.context);
    }

    @Override
    public GpuMatrix backward(GpuMatrix outputGradient, Object context, float learningRate) {
        GpuMatrix sublayerGrad = sublayer.backward(outputGradient, context, learningRate);
        GpuMatrix inputGrad = sublayerGrad.add(outputGradient); // Allocates a new GpuMatrix
        sublayerGrad.close(); // Free the intermediate gradient
        return inputGrad;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {
        sublayer.save(dos);
    }

    @Override
    public void load(DataInputStream dis) throws IOException {
        sublayer.load(dis);
    }

    @Override
    public void close() {
        try {
            sublayer.close();
        } catch (Exception e) {
            throw new RuntimeException("Failed to close sublayer inside GpuResidualBlock: " + e.getMessage(), e);
        }
    }
}
