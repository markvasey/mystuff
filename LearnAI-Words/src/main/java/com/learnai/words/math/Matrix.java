package com.learnai.words.math;

import java.util.concurrent.ThreadLocalRandom;

public class Matrix {
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
        double variance = Math.sqrt(2.0 / (rows + cols));
        ThreadLocalRandom r = ThreadLocalRandom.current();
        for (int i = 0; i < m.data.length; i++) {
            m.data[i] = r.nextGaussian() * variance;
        }
        return m;
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public double[] getData() { return data; }

    public double get(int r, int c) {
        return data[r * cols + c];
    }

    public void set(int r, int c, double val) {
        data[r * cols + c] = val;
    }

    public Matrix multiply(Matrix other) {
        return multiply(other, false, false);
    }

    public Matrix multiply(Matrix other, boolean transThis, boolean transOther) {
        int leftRows = transThis ? this.cols : this.rows;
        int leftCols = transThis ? this.rows : this.cols;
        int rightRows = transOther ? other.cols : other.rows;
        int rightCols = transOther ? other.rows : other.cols;

        if (leftCols != rightRows) {
            throw new IllegalArgumentException("Incompatible dimensions: " + leftCols + " != " + rightRows);
        }

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
                    int kOff = k * rightCols;
                    for (int j = 0; j < rightCols; j++) {
                        rData[rOff + j] += val * bData[kOff + j];
                    }
                }
            }
        } else if (transThis && !transOther) {
            for (int k = 0; k < leftCols; k++) {
                int kOffA = k * leftRows;
                int kOffB = k * rightCols;
                for (int i = 0; i < leftRows; i++) {
                    double val = aData[kOffA + i];
                    int iOffR = i * rightCols;
                    for (int j = 0; j < rightCols; j++) {
                        rData[iOffR + j] += val * bData[kOffB + j];
                    }
                }
            }
        } else if (!transThis && transOther) {
            for (int i = 0; i < leftRows; i++) {
                int iOff = i * leftCols;
                int rOff = i * rightCols;
                for (int j = 0; j < rightCols; j++) {
                    int jOff = j * leftCols;
                    double sum = 0;
                    for (int k = 0; k < leftCols; k++) {
                        sum += aData[iOff + k] * bData[jOff + k];
                    }
                    rData[rOff + j] = sum;
                }
            }
        }
        return res;
    }

    public Matrix add(Matrix other) {
        if (this.cols != other.cols) throw new IllegalArgumentException("Cols mismatch");
        Matrix result = new Matrix(rows, cols);
        double[] r = result.data;
        double[] a = this.data;
        double[] b = other.data;
        if (other.rows == 1) {
            for (int i = 0; i < rows; i++) {
                int off = i * cols;
                for (int j = 0; j < cols; j++) r[off + j] = a[off + j] + b[j];
            }
        } else if (this.rows == other.rows) {
            for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        } else {
            throw new IllegalArgumentException("Rows mismatch");
        }
        return result;
    }

    public void addInPlace(Matrix other) {
        double[] a = this.data;
        double[] b = other.data;
        if (other.rows == 1) {
            for (int i = 0; i < rows; i++) {
                int off = i * cols;
                for (int j = 0; j < cols; j++) a[off + j] += b[j];
            }
        } else {
            for (int i = 0; i < a.length; i++) a[i] += b[i];
        }
    }

    public void subtractInPlace(Matrix other) {
        double[] a = this.data;
        double[] b = other.data;
        for (int i = 0; i < a.length; i++) a[i] -= b[i];
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, this.get(i, j));
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Dim mismatch");
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) res.data[i] = this.data[i] - other.data[i];
        return res;
    }

    public Matrix multiplyElementWise(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) throw new IllegalArgumentException("Dim mismatch");
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) res.data[i] = this.data[i] * other.data[i];
        return res;
    }

    public Matrix square() {
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) res.data[i] = this.data[i] * this.data[i];
        return res;
    }

    public Matrix sqrt(double epsilon) {
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < data.length; i++) res.data[i] = Math.sqrt(this.data[i] + epsilon);
        return res;
    }

    public Matrix rowMean() {
        Matrix res = new Matrix(rows, 1);
        double[] rData = res.data;
        double[] mData = this.data;
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            int off = i * cols;
            for (int j = 0; j < cols; j++) sum += mData[off + j];
            rData[i] = sum / cols;
        }
        return res;
    }

    public Matrix rowVariance(Matrix mean) {
        Matrix res = new Matrix(rows, 1);
        double[] rData = res.data;
        double[] mData = this.data;
        double[] meanData = mean.data;
        for (int i = 0; i < rows; i++) {
            double sumSq = 0;
            double m = meanData[i];
            int off = i * cols;
            for (int j = 0; j < cols; j++) {
                double diff = mData[off + j] - m;
                sumSq += diff * diff;
            }
            rData[i] = sumSq / cols;
        }
        return res;
    }

    public void save(java.io.DataOutputStream dos) throws java.io.IOException {
        dos.writeInt(rows);
        dos.writeInt(cols);
        for (double d : data) {
            dos.writeDouble(d);
        }
    }

    public static Matrix load(java.io.DataInputStream dis) throws java.io.IOException {
        int r = dis.readInt();
        int c = dis.readInt();
        Matrix m = new Matrix(r, c);
        for (int i = 0; i < m.data.length; i++) {
            m.data[i] = dis.readDouble();
        }
        return m;
    }
}
