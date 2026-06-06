package com.learnai.math;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Fundamental Matrix class for linear algebra operations.
 * Built from scratch for neural network computations.
 */
public class Matrix {
    private final int rows;
    private final int cols;
    private final double[][] data;
    private static final Random random = new Random();

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = data;
    }

    public static Matrix random(int rows, int cols) {
        Matrix m = new Matrix(rows, cols);
        // Xavier/Glorot initialization for better neural network convergence
        double limit = Math.sqrt(6.0 / (rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m.data[i][j] = (random.nextDouble() * 2 - 1) * limit;
            }
        }
        return m;
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public double get(int r, int c) { return data[r][c]; }
    public void set(int r, int c, double val) { data[r][c] = val; }

    public Matrix dot(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions mismatch for dot product: " + this.cols + " vs " + other.rows);
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    public Matrix add(Matrix other) {
        if (this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix column dimensions mismatch for addition: " + this.cols + " vs " + other.cols);
        }
        if (other.rows != 1 && other.rows != this.rows) {
            throw new IllegalArgumentException("Matrix row dimensions mismatch for addition: " + this.rows + " vs " + other.rows);
        }
        
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            int otherRow = (other.rows == 1) ? 0 : i;
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[otherRow][j];
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        if (this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix column dimensions mismatch for subtraction: " + this.cols + " vs " + other.cols);
        }
        if (other.rows != 1 && other.rows != this.rows) {
            throw new IllegalArgumentException("Matrix row dimensions mismatch for subtraction: " + this.rows + " vs " + other.rows);
        }

        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            int otherRow = (other.rows == 1) ? 0 : i;
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] - other.data[otherRow][j];
            }
        }
        return result;
    }

    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * scalar;
            }
        }
        return result;
    }

    public Matrix multiplyElementWise(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions mismatch for element-wise multiplication");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    public Matrix apply(DoubleUnaryOperator func) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func.applyAsDouble(this.data[i][j]);
            }
        }
        return result;
    }

    public Matrix sumRows() {
        Matrix result = new Matrix(1, cols);
        for (int j = 0; j < cols; j++) {
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                sum += data[i][j];
            }
            result.data[0][j] = sum;
        }
        return result;
    }

    public double sum() {
        double sum = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                sum += data[i][j];
            }
        }
        return sum;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < Math.min(rows, 5); i++) {
            for (int j = 0; j < Math.min(cols, 5); j++) {
                sb.append(String.format("%8.4f ", data[i][j]));
            }
            if (cols > 5) sb.append("...");
            sb.append("\n");
        }
        if (rows > 5) sb.append("...\n");
        return sb.toString();
    }
}
