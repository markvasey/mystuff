package com.tapoviewer.model;

import org.bytedeco.opencv.opencv_core.Mat;
import org.jtransforms.fft.DoubleFFT_1D;

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

    /**
     * Cached JTransforms FFT instance (Priority 4).
     * DoubleFFT_1D is not thread-safe for concurrent calls on the same instance,
     * but each TrackedPerson is accessed from exactly one thread.
     */
    private final DoubleFFT_1D fft;

    private static final double MIN_SEIZURE_MOTION = 50.0; // Noise floor for FFT magnitude
    private static final double DOMINANCE_THRESHOLD = 0.40; // Target band must have at least 40% of AC power

    // Two-stage seizure validation fields
    private final java.util.Queue<Boolean> warningHistory = new java.util.LinkedList<>();
    private final int densityWindowSize;
    private final int confirmationThresholdFrames;
    private int warningFrameCount = 0;

    private boolean seizureWarning = false;
    private boolean seizureConfirmed = false;
    private boolean seizureDetected = false;
    private boolean tonicWarning = false;

    private int consecutiveSeizureFrames = 0;
    private int consecutiveTonicFrames = 0;
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
            this.densityWindowSize = 150; // 15 seconds @ 10 fps
            this.confirmationThresholdFrames = 5; // 0.5 seconds
        } else {
            this.historySize = 64;
            this.resolutionHz = 0.3125; // 20 / 64 = 0.3125 Hz per bin
            this.minBin = 6;
            this.maxBin = 15;
            this.densityWindowSize = 300; // 15 seconds @ 20 fps
            this.confirmationThresholdFrames = 10; // 0.5 seconds
        }
        this.fft = new DoubleFFT_1D(historySize);
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
            seizureWarning = false;
            seizureConfirmed = false;
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
            // 2. Perform FFT in-place using JTransforms (Priority 4).
            // realForward() modifies data[] in-place with packed output:
            //   data[0]      = Re[0]    (DC component, skipped)
            //   data[1]      = Re[N/2]  (Nyquist, skipped)
            //   data[2*k]    = Re[k]    for k = 1 .. N/2-1
            //   data[2*k+1]  = Im[k]    for k = 1 .. N/2-1
            // No Complex[] allocation — 2-4x faster than ACM for N=32/64.
            fft.realForward(data);

            // 3. Compute power spectrum of AC components (ignoring DC at index 0)
            double totalPower = 0.0;
            double targetBandPower = 0.0;
            double maxAmp = 0.0;
            int peakBin = 0;

            double tonicBandPower = 0.0;
            double maxTonicAmp = 0.0;
            int minTonicBin = frameSkipping ? 11 : 19; // 3.5 Hz @ 10fps, 6.0 Hz @ 20fps
            int maxTonicBin = frameSkipping ? 16 : (historySize / 2 - 1); // 5.0 Hz @ 10fps, Nyquist @ 20fps

            for (int i = 1; i < historySize / 2; i++) {
                // Magnitude of bin i from packed JTransforms output
                double amp = Math.hypot(data[2 * i], data[2 * i + 1]);
                totalPower += amp;

                if (i >= minBin && i <= maxBin) {
                    targetBandPower += amp;
                    if (amp > maxAmp) {
                        maxAmp = amp;
                        peakBin = i;
                    }
                }

                if (i >= minTonicBin && i <= maxTonicBin) {
                    tonicBandPower += amp;
                    if (amp > maxTonicAmp) {
                        maxTonicAmp = amp;
                    }
                }
            }

            // Calculate peak frequency in Hz
            this.peakFrequencyHz = peakBin * resolutionHz;

            // If frame skipping is enabled, scale the standard 32-point FFT values by 2.0
            // so they match the magnitude/power thresholds calibrated for the 64-point FFT.
            if (frameSkipping) {
                maxAmp *= 2.0;
                maxTonicAmp *= 2.0;
                totalPower *= 2.0;
                targetBandPower *= 2.0;
                tonicBandPower *= 2.0;
            }

            this.peakAmplitude = maxAmp;
            this.totalPower = totalPower;

            // Calculate Peak-to-Average Power Ratio (PAPR) to identify narrow-band oscillations
            double averageAC = totalPower / (historySize / 2 - 1);
            this.papr = (averageAC > 0.001) ? (maxAmp / averageAC) : 0.0;
            
            double tonicPapr = (averageAC > 0.001) ? (maxTonicAmp / averageAC) : 0.0;

            double dominanceThreshold = frameSkipping ? 0.30 : 0.40;
            double paprThreshold = frameSkipping ? 2.2 : 2.5;

            if (totalPower > MIN_SEIZURE_MOTION) {
                double dominance = targetBandPower / totalPower;
                double tonicDominance = tonicBandPower / totalPower;
                
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

                boolean currentFrameTonic = (tonicDominance >= (frameSkipping ? 0.25 : 0.30) && maxTonicAmp > (frameSkipping ? 8.0 : 10.0) && tonicPapr > (frameSkipping ? 2.0 : 2.2) && aspectRatio <= 1.8);
                if (currentFrameTonic) {
                    consecutiveTonicFrames++;
                } else {
                    consecutiveTonicFrames = 0;
                }
                
                // Stage 1 (Warning): Requires 2 consecutive frames of clonic or tonic indicators
                boolean clonicActive = (consecutiveSeizureFrames >= 2);
                boolean tonicActive = (consecutiveTonicFrames >= 2);
                
                this.seizureWarning = clonicActive || tonicActive;
                this.tonicWarning = tonicActive;
                
                // Slide the warning history queue - only clonic warnings accumulate for confirmed alarm
                warningHistory.add(clonicActive);
                if (clonicActive) {
                    warningFrameCount++;
                }
                if (warningHistory.size() > densityWindowSize) {
                    boolean removed = warningHistory.poll();
                    if (removed) {
                        warningFrameCount--;
                    }
                }
                
                // Stage 2 (Confirmed): Warning must be active for >= 0.5 seconds cumulatively
                this.seizureConfirmed = (warningFrameCount >= confirmationThresholdFrames);
                this.seizureDetected = this.seizureConfirmed; // For backwards compatibility
            } else {
                consecutiveSeizureFrames = 0;
                consecutiveTonicFrames = 0;
                seizureWarning = false;
                tonicWarning = false;
                
                // Also push false to history to fade out warning count when person stops moving
                warningHistory.add(false);
                if (warningHistory.size() > densityWindowSize) {
                    boolean removed = warningHistory.poll();
                    if (removed) {
                        warningFrameCount--;
                    }
                }
                
                this.seizureConfirmed = (warningFrameCount >= confirmationThresholdFrames);
                this.seizureDetected = this.seizureConfirmed;
            }

        } catch (Exception e) {
            System.err.println("TrackedPerson: FFT analysis failed: " + e.getMessage());
            seizureWarning = false;
            tonicWarning = false;
            seizureConfirmed = false;
            seizureDetected = false;
        }
    }

    public boolean isSeizureDetected() { return seizureConfirmed; }
    public boolean isSeizureWarning() { return seizureWarning; }
    public boolean isSeizureConfirmed() { return seizureConfirmed; }
    public boolean isTonicWarning() { return tonicWarning; }
    
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
    public boolean isDetectedInCurrentFrame() { return lastSeenFrames == 0; }
    
    public void release() {
        if (lastGrayRegion != null) {
            lastGrayRegion.release();
            lastGrayRegion = null;
        }
    }
}
