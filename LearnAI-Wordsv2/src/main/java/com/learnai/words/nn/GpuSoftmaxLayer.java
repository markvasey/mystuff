package com.learnai.words.nn;

import com.learnai.words.math.GpuMatrix;
import com.learnai.words.math.CudaBridge;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class GpuSoftmaxLayer implements GpuLayer {

    @Override
    public GpuForwardResult forward(GpuMatrix input) {
        GpuMatrix output = new GpuMatrix(input.getRows(), input.getCols());
        CudaBridge.cudaSoftmaxForward(input.getDevicePtr(), output.getDevicePtr(), input.getRows(), input.getCols());
        
        GpuMatrix outputCopy = new GpuMatrix(input.getRows(), input.getCols());
        long byteSize = (long) input.getRows() * input.getCols() * java.lang.foreign.ValueLayout.JAVA_FLOAT.byteSize();
        CudaBridge.cudaMemcpyToDevice(outputCopy.getDevicePtr(), output.getDevicePtr(), byteSize);

        return new GpuForwardResult(output, outputCopy);
    }

    @Override
    public GpuMatrix backward(GpuMatrix target, Object context, float learningRate) {
        GpuMatrix lastOutput = (GpuMatrix) context;
        GpuMatrix gradient = lastOutput.subtract(target);
        lastOutput.close();
        return gradient;
    }

    @Override
    public void save(DataOutputStream dos) throws IOException {}

    @Override
    public void load(DataInputStream dis) throws IOException {}

    @Override
    public void close() {}
}
