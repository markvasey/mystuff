package com.learnai.words.math;

import java.util.concurrent.ThreadLocalRandom;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public class Matrix {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;
    private final int rows;
    private final int cols;
    private final double[] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    public static Matrix random(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        // Ultra-conservative initialization for deep networks
        double scale = 0.01;
        ThreadLocalRandom r = ThreadLocalRandom.current();
        for (int i = 0; i < m.data.length; i++) {
            m.data[i] = r.nextGaussian() * scale;
        }
        return m;
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public double[] getData() { return data; }
    public double get(int r, int c) { return data[r * cols + c]; }
    public void set(int r, int c, double val) { data[r * cols + c] = val; }

    public Matrix multiply(Matrix other) { return multiply(other, false, false); }

    public Matrix multiply(Matrix other, boolean transThis, boolean transOther) {
        int leftRows = transThis ? this.cols : this.rows;
        int leftCols = transThis ? this.rows : this.cols;
        int rightRows = transOther ? other.cols : other.rows;
        int rightCols = transOther ? other.rows : other.cols;

        if (leftCols != rightRows) throw new IllegalArgumentException("Dim mismatch: " + leftCols + " != " + rightRows);

        Matrix res = new Matrix(leftRows, rightCols);
        double[] rData = res.data;
        double[] aData = this.data;
        double[] bData = other.data;

        if (!transThis && !transOther) {
            for (int i = 0; i < leftRows; i++) {
                int iOff = i * leftCols;
                int rOff = i * rightCols;
                for (int k = 0; k < leftCols; k++) {
                    double val = aData[iOff + k];
                    if (val == 0) continue;
                    int kOff = k * rightCols;
                    int j = 0;
                    for (; j < SPECIES.loopBound(rightCols); j += SPECIES.length()) {
                        var vb = DoubleVector.fromArray(SPECIES, bData, kOff + j);
                        var vr = DoubleVector.fromArray(SPECIES, rData, rOff + j);
                        vb.broadcast(val).add(vr).intoArray(rData, rOff + j);
                    }
                    for (; j < rightCols; j++) rData[rOff + j] += val * bData[kOff + j];
                }
            }
        } else if (transThis && !transOther) {
            for (int k = 0; k < leftCols; k++) {
                int kOffA = k * leftRows;
                int kOffB = k * rightCols;
                for (int i = 0; i < leftRows; i++) {
                    double val = aData[kOffA + i];
                    if (val == 0) continue;
                    int iOffR = i * rightCols;
                    int j = 0;
                    for (; j < SPECIES.loopBound(rightCols); j += SPECIES.length()) {
                        var vb = DoubleVector.fromArray(SPECIES, bData, kOffB + j);
                        var vr = DoubleVector.fromArray(SPECIES, rData, iOffR + j);
                        vb.broadcast(val).add(vr).intoArray(rData, iOffR + j);
                    }
                    for (; j < rightCols; j++) rData[iOffR + j] += val * bData[kOffB + j];
                }
            }
        } else if (!transThis && transOther) {
            // For (!transThis && transOther), it's more efficient to transpose 'other' once
            // and use the standard vectorized path, rather than using the dot-product path.
            Matrix otherT = other.transpose();
            return this.multiply(otherT, false, false);
        }
        return res;
    }

    /** SIMD Vectorized Add In-Place with fixed broadcasting index */
    public void addInPlace(Matrix other) {
        double[] a = this.data;
        double[] b = other.data;
        if (other.rows == 1 && this.rows > 1) { // Broadcasting
            for (int i = 0; i < rows; i++) {
                int off = i * cols;
                int j = 0;
                for (; j < SPECIES.loopBound(cols); j += SPECIES.length()) {
                    var va = DoubleVector.fromArray(SPECIES, a, off + j);
                    var vb = DoubleVector.fromArray(SPECIES, b, j);
                    va.add(vb).intoArray(a, off + j);
                }
                for (; j < cols; j++) a[off + j] += b[j];
            }
        } else {
            int i = 0;
            for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
                var va = DoubleVector.fromArray(SPECIES, a, i);
                var vb = DoubleVector.fromArray(SPECIES, b, i);
                va.add(vb).intoArray(a, i);
            }
            for (; i < a.length; i++) a[i] += b[i];
        }
    }

    /** SIMD Vectorized Subtract In-Place */
    public void subtractInPlace(Matrix other) {
        double[] a = this.data;
        double[] b = other.data;
        int i = 0;
        for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            va.sub(vb).intoArray(a, i);
        }
        for (; i < a.length; i++) a[i] -= b[i];
    }

    public Matrix add(Matrix other) {
        Matrix res = new Matrix(rows, cols);
        System.arraycopy(this.data, 0, res.data, 0, data.length);
        res.addInPlace(other);
        return res;
    }

    public Matrix subtract(Matrix other) {
        Matrix res = new Matrix(rows, cols);
        System.arraycopy(this.data, 0, res.data, 0, data.length);
        res.subtractInPlace(other);
        return res;
    }

    public Matrix multiplyElementWise(Matrix other) {
        Matrix res = new Matrix(rows, cols);
        double[] a = this.data;
        double[] b = other.data;
        double[] r = res.data;
        int i = 0;
        for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            var vb = DoubleVector.fromArray(SPECIES, b, i);
            va.mul(vb).intoArray(r, i);
        }
        for (; i < a.length; i++) r[i] = a[i] * b[i];
        return res;
    }

    public Matrix square() {
        Matrix res = new Matrix(rows, cols);
        double[] a = this.data;
        double[] r = res.data;
        int i = 0;
        for (; i < SPECIES.loopBound(a.length); i += SPECIES.length()) {
            var va = DoubleVector.fromArray(SPECIES, a, i);
            va.mul(va).intoArray(r, i);
        }
        for (; i < a.length; i++) r[i] = a[i] * a[i];
        return res;
    }

    public Matrix sqrt(double epsilon) {
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) res.data[i] = Math.sqrt(this.data[i] + epsilon);
        return res;
    }

    public Matrix rowMean() {
        Matrix res = new Matrix(rows, 1);
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            int off = i * cols;
            for (int j = 0; j < cols; j++) sum += data[off + j];
            res.data[i] = sum / cols;
        }
        return res;
    }

    public Matrix rowVariance(Matrix mean) {
        Matrix res = new Matrix(rows, 1);
        for (int i = 0; i < rows; i++) {
            double sumSq = 0;
            double m = mean.data[i];
            int off = i * cols;
            for (int j = 0; j < cols; j++) {
                double diff = data[off + j] - m;
                sumSq += diff * diff;
            }
            res.data[i] = sumSq / cols;
        }
        return res;
    }

    public Matrix transpose() {
        Matrix res = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) res.data[j * rows + i] = data[i * cols + j];
        }
        return res;
    }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        dos.writeInt(rows); dos.writeInt(cols);
        for (double d : data) dos.writeDouble(d);
    }

    public static Matrix load(java.io.DataInputStream dis) throws java.io.IOException {
        int r = dis.readInt(); int c = dis.readInt();
        Matrix m = new Matrix(r, c);
        for (int i = 0; i < m.data.length; i++) m.data[i] = dis.readDouble();
        return m;
    }

    public boolean hasInvalidValues() {
        for (double d : data) {
            if (Double.isNaN(d) || Double.isInfinite(d)) return true;
        }
        return false;
    }
}
