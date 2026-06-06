package com.learnai.words.math;

import java.util.Random;

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
        Random r = new Random();
        double variance = Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < m.data.length; i++) {
            m.data[i] = r.nextGaussian() * variance;
        }
        return m;
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }

    public double get(int r, int c) {
        return data[r * cols + c];
    }

    public void set(int r, int c, double val) {
        data[r * cols + c] = val;
    }

    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible dimensions: " + this.cols + " != " + other.rows);
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int k = 0; k < this.cols; k++) {
                double val = this.get(i, k);
                for (int j = 0; j < other.cols; j++) {
                    result.data[i * other.cols + j] += val * other.get(k, j);
                }
            }
        }
        return result;
    }

    public Matrix add(Matrix other) {
        if (this.cols != other.cols) {
            throw new IllegalArgumentException("Incompatible dimensions: columns must match");
        }
        Matrix result = new Matrix(rows, cols);
        if (other.rows == 1) {
            // Broadcasting: add single row to all rows of this matrix
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    result.set(i, j, this.get(i, j) + other.get(0, j));
                }
            }
        } else if (this.rows == other.rows) {
            for (int i = 0; i < data.length; i++) {
                result.data[i] = this.data[i] + other.data[i];
            }
        } else {
            throw new IllegalArgumentException("Incompatible dimensions: rows must match or other must be a single row");
        }
        return result;
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
    
    public double[] getData() {
        return data;
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
        for (int i = 0; i < rows; i++) {
            double sum = 0;
            for (int j = 0; j < cols; j++) sum += get(i, j);
            res.set(i, 0, sum / cols);
        }
        return res;
    }

    public Matrix rowVariance(Matrix mean) {
        Matrix res = new Matrix(rows, 1);
        for (int i = 0; i < rows; i++) {
            double sumSq = 0;
            double m = mean.get(i, 0);
            for (int j = 0; j < cols; j++) {
                double diff = get(i, j) - m;
                sumSq += diff * diff;
            }
            res.set(i, 0, sumSq / cols);
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
