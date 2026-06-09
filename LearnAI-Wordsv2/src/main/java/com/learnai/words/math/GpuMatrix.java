package com.learnai.words.math;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.ref.Cleaner;

public class GpuMatrix implements AutoCloseable {
    private static final Cleaner CLEANER = Cleaner.create();

    private final int rows;
    private final int cols;
    private final MemorySegment devicePtr;
    private final Cleaner.Cleanable cleanable;

    private static class CleanupState implements Runnable {
        private final MemorySegment ptr;

        public CleanupState(MemorySegment ptr) {
            this.ptr = ptr;
        }

        @Override
        public void run() {
            if (ptr != null && !ptr.equals(MemorySegment.NULL)) {
                CudaBridge.cudaFree(ptr);
            }
        }
    }

    public GpuMatrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        long byteSize = (long) rows * cols * ValueLayout.JAVA_FLOAT.byteSize();
        this.devicePtr = CudaBridge.cudaMalloc(byteSize);
        this.cleanable = CLEANER.register(this, new CleanupState(devicePtr));
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public MemorySegment getDevicePtr() {
        return devicePtr;
    }

    public void upload(float[] hostData) {
        if (hostData.length != rows * cols) {
            throw new IllegalArgumentException("Length mismatch: hostData length (" + hostData.length +
                    ") != matrix size (" + (rows * cols) + ")");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment hostSrc = arena.allocateFrom(ValueLayout.JAVA_FLOAT, hostData);
            long byteSize = (long) hostData.length * ValueLayout.JAVA_FLOAT.byteSize();
            CudaBridge.cudaMemcpyToDevice(devicePtr, hostSrc, byteSize);
        }
    }

    public void download(float[] dest) {
        if (dest.length != rows * cols) {
            throw new IllegalArgumentException("Length mismatch: dest length (" + dest.length +
                    ") != matrix size (" + (rows * cols) + ")");
        }
        try (Arena arena = Arena.ofConfined()) {
            long byteSize = (long) dest.length * ValueLayout.JAVA_FLOAT.byteSize();
            MemorySegment hostDest = arena.allocate(ValueLayout.JAVA_FLOAT, dest.length);
            CudaBridge.cudaMemcpyToHost(hostDest, devicePtr, byteSize);
            MemorySegment.copy(hostDest, ValueLayout.JAVA_FLOAT, 0, dest, 0, dest.length);
        }
    }

    public GpuMatrix multiply(GpuMatrix other) {
        return multiply(other, false, false);
    }

    public GpuMatrix multiply(GpuMatrix other, boolean transThis, boolean transOther) {
        int leftRows = transThis ? this.cols : this.rows;
        int leftCols = transThis ? this.rows : this.cols;
        int rightRows = transOther ? other.cols : other.rows;
        int rightCols = transOther ? other.rows : other.cols;

        if (leftCols != rightRows) {
            throw new IllegalArgumentException("Dimension mismatch for matrix multiplication: " +
                    leftCols + " != " + rightRows);
        }

        GpuMatrix res = new GpuMatrix(leftRows, rightCols);
        CudaBridge.cudaMatrixMultiply(this.devicePtr, other.devicePtr, res.devicePtr,
                leftRows, rightCols, leftCols, transThis, transOther);
        return res;
    }

    public void addInPlace(GpuMatrix other) {
        CudaBridge.cudaAddInPlace(this.devicePtr, other.devicePtr, this.rows, this.cols, other.rows, other.cols);
    }

    public void subtractInPlace(GpuMatrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Dimension mismatch for subtractInPlace: (" +
                    this.rows + "x" + this.cols + ") != (" + other.rows + "x" + other.cols + ")");
        }
        CudaBridge.cudaSubtractInPlace(this.devicePtr, other.devicePtr, this.rows * this.cols);
    }

    public GpuMatrix add(GpuMatrix other) {
        GpuMatrix res = new GpuMatrix(this.rows, this.cols);
        try (Arena arena = Arena.ofConfined()) {
            long byteSize = (long) this.rows * this.cols * ValueLayout.JAVA_FLOAT.byteSize();
            CudaBridge.cudaMemcpyToDevice(res.devicePtr, this.devicePtr, byteSize);
        }
        res.addInPlace(other);
        return res;
    }

    public GpuMatrix subtract(GpuMatrix other) {
        GpuMatrix res = new GpuMatrix(this.rows, this.cols);
        try (Arena arena = Arena.ofConfined()) {
            long byteSize = (long) this.rows * this.cols * ValueLayout.JAVA_FLOAT.byteSize();
            CudaBridge.cudaMemcpyToDevice(res.devicePtr, this.devicePtr, byteSize);
        }
        res.subtractInPlace(other);
        return res;
    }

    public GpuMatrix multiplyElementWise(GpuMatrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Dimension mismatch for multiplyElementWise: (" +
                    this.rows + "x" + this.cols + ") != (" + other.rows + "x" + other.cols + ")");
        }
        GpuMatrix res = new GpuMatrix(this.rows, this.cols);
        CudaBridge.cudaMultiplyElementWise(this.devicePtr, other.devicePtr, res.devicePtr, this.rows * this.cols);
        return res;
    }

    public GpuMatrix square() {
        GpuMatrix res = new GpuMatrix(this.rows, this.cols);
        CudaBridge.cudaSquare(this.devicePtr, res.devicePtr, this.rows * this.cols);
        return res;
    }

    public GpuMatrix sqrt(float epsilon) {
        GpuMatrix res = new GpuMatrix(this.rows, this.cols);
        CudaBridge.cudaSqrt(this.devicePtr, res.devicePtr, this.rows * this.cols, epsilon);
        return res;
    }

    public GpuMatrix rowMean() {
        GpuMatrix res = new GpuMatrix(this.rows, 1);
        CudaBridge.cudaRowMean(this.devicePtr, res.devicePtr, this.rows, this.cols);
        return res;
    }

    public GpuMatrix rowVariance(GpuMatrix mean) {
        if (mean.getRows() != this.rows || mean.getCols() != 1) {
            throw new IllegalArgumentException("Mean matrix must be (" + this.rows + "x1) dimension");
        }
        GpuMatrix res = new GpuMatrix(this.rows, 1);
        CudaBridge.cudaRowVariance(this.devicePtr, mean.getDevicePtr(), res.devicePtr, this.rows, this.cols);
        return res;
    }

    public GpuMatrix transpose() {
        GpuMatrix res = new GpuMatrix(this.cols, this.rows);
        CudaBridge.cudaTranspose(this.devicePtr, res.devicePtr, this.rows, this.cols);
        return res;
    }

    public Matrix toCpu() {
        Matrix m = new Matrix(this.rows, this.cols);
        this.download(m.getData());
        return m;
    }

    public static GpuMatrix fromCpu(Matrix cpuMat) {
        GpuMatrix m = new GpuMatrix(cpuMat.getRows(), cpuMat.getCols());
        m.upload(cpuMat.getData());
        return m;
    }

    @Override
    public void close() {
        cleanable.clean();
    }
}
