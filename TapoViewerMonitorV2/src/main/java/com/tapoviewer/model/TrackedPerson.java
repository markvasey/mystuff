package com.tapoviewer.model;

import org.bytedeco.opencv.opencv_core.Mat;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.TransformType;
import org.apache.commons.math3.complex.Complex;

import java.awt.Rectangle;
import java.util.LinkedList;

public class TrackedPerson {
    private Rectangle bounds;
    private Mat lastGrayRegion;
    
    // Joint coordinate history
    private float[][] lastKeypoints; // [17][3] (x, y, conf)
    
    // Unified motion signal history
    private final LinkedList<Double> motionHistory = new LinkedList<>();
    private final int historySize;
    private final double resolutionHz;
    private final int minBin;
    private final int maxBin;
    private final boolean frameSkipping;

    private static final double MIN_SEIZURE_MOTION = 50.0; // Noise floor for FFT magnitude
    private static final double DOMINANCE_THRESHOLD = 0.40; // Target band must have at least 40% of AC power

    private boolean seizureDetected = false;
    private int consecutiveSeizureFrames = 0;
    private int lastSeenFrames = 0;
    private double cumulativeMotion = 0;
    
    // Real-time tracking frequency log
    private double peakFrequencyHz = 0.0;
    private double peakAmplitude = 0.0;
    private double totalPower = 0.0;
    private double papr = 0.0;

    public TrackedPerson(Rectangle bounds) {
        this(bounds, false);
    }

    public TrackedPerson(Rectangle bounds, boolean frameSkipping) {
        this.bounds = bounds;
        this.frameSkipping = frameSkipping;
        if (frameSkipping) {
            this.historySize = 32;
            this.resolutionHz = 0.3125; // 10 / 32 = 0.3125 Hz per bin
            this.minBin = 6;            // 2.0 / 0.3125 = 6.4 (bin 6)
            this.maxBin = 14;           // 4.5 / 0.3125 = 14.4 (bin 14)
        } else {
            this.historySize = 64;
            this.resolutionHz = 0.3125; // 20 / 64 = 0.3125 Hz per bin
            this.minBin = 6;
            this.maxBin = 15;
        }
    }

    public Rectangle getBounds() { return bounds; }
    public void setBounds(Rectangle bounds) { this.bounds = bounds; }

    public Mat getLastGrayRegion() { return lastGrayRegion; }
    public void setLastGrayRegion(Mat lastGrayRegion) {
        if (this.lastGrayRegion != null) {
            this.lastGrayRegion.release();
        }
        this.lastGrayRegion = lastGrayRegion;
    }

    private float[][] prevKeypoints;
    public float[][] getPrevKeypoints() { return prevKeypoints; }
    public float[][] getLastKeypoints() { return lastKeypoints; }
    public void setLastKeypoints(float[][] keypoints) {
        this.prevKeypoints = this.lastKeypoints;
        this.lastKeypoints = keypoints;
    }

    public void addMotion(double magnitude) {
        motionHistory.add(magnitude);
        cumulativeMotion += magnitude;
        if (motionHistory.size() > historySize) {
            motionHistory.removeFirst();
        }
        analyzeSeizure();
    }

    public double getLastMotionValue() {
        return motionHistory.isEmpty() ? 0.0 : motionHistory.getLast();
    }

    private void analyzeSeizure() {
        if (motionHistory.size() < historySize) {
            seizureDetected = false;
            return;
        }

        // 1. Prepare data for FFT
        double[] data = new double[historySize];
        int idx = 0;
        for (Double val : motionHistory) {
            data[idx++] = val;
        }

        try {
            // 2. Perform Fast Fourier Transform (FFT)
            FastFourierTransformer transformer = new FastFourierTransformer(DftNormalization.STANDARD);
            Complex[] fftResult = transformer.transform(data, TransformType.FORWARD);

            // 3. Compute power spectrum of AC components (ignoring DC at index 0)
            double totalPower = 0.0;
            double targetBandPower = 0.0;
            double maxAmp = 0.0;
            int peakBin = 0;

            for (int i = 1; i < historySize / 2; i++) {
                double amp = fftResult[i].abs();
                totalPower += amp;

                if (i >= minBin && i <= maxBin) {
                    targetBandPower += amp;
                    if (amp > maxAmp) {
                        maxAmp = amp;
                        peakBin = i;
                    }
                }
            }

            // Calculate peak frequency in Hz
            this.peakFrequencyHz = peakBin * resolutionHz;

            // If frame skipping is enabled, scale the standard 32-point FFT values by 2.0
            // so they match the magnitude/power thresholds calibrated for the 64-point FFT.
            if (frameSkipping) {
                maxAmp *= 2.0;
                totalPower *= 2.0;
                targetBandPower *= 2.0;
            }

            this.peakAmplitude = maxAmp;
            this.totalPower = totalPower;

            // Calculate Peak-to-Average Power Ratio (PAPR) to identify narrow-band oscillations
            double averageAC = totalPower / (historySize / 2 - 1);
            this.papr = (averageAC > 0.001) ? (maxAmp / averageAC) : 0.0;

            double dominanceThreshold = frameSkipping ? 0.30 : 0.40;
            double paprThreshold = frameSkipping ? 2.2 : 2.5;

            if (totalPower > MIN_SEIZURE_MOTION) {
                double dominance = targetBandPower / totalPower;
                
                // Calculate aspect ratio (height / width) to detect standing posture
                double aspectRatio = 0.0;
                if (bounds != null && bounds.width > 0) {
                    aspectRatio = (double) bounds.height / bounds.width;
                }
                
                boolean currentFrameSeizure = (dominance >= dominanceThreshold && maxAmp > 25.0 && this.papr > paprThreshold && aspectRatio <= 1.8);
                if (currentFrameSeizure) {
                    consecutiveSeizureFrames++;
                } else {
                    consecutiveSeizureFrames = 0;
                }
                
                // Flag seizure if it persists for at least 2 consecutive frames
                seizureDetected = (consecutiveSeizureFrames >= 2);
            } else {
                consecutiveSeizureFrames = 0;
                seizureDetected = false;
            }

        } catch (Exception e) {
            System.err.println("TrackedPerson: FFT analysis failed: " + e.getMessage());
            seizureDetected = false;
        }
    }

    public boolean isSeizureDetected() { return seizureDetected; }
    
    public double getPeakFrequencyHz() { return peakFrequencyHz; }
    public double getPeakAmplitude() { return peakAmplitude; }
    public double getTotalPower() { return totalPower; }
    public double getPapr() { return papr; }

    public boolean isLikelyStaticObject() {
        // Objects with low lifetime movement are ignored as furniture
        return (lastSeenFrames > 50 && (cumulativeMotion / (lastSeenFrames + 1)) < 0.2);
    }
    
    public int incrementLastSeen() { return ++lastSeenFrames; }
    public void resetLastSeen() { lastSeenFrames = 0; }
    
    public void release() {
        if (lastGrayRegion != null) {
            lastGrayRegion.release();
            lastGrayRegion = null;
        }
    }
}
