package com.tapoviewer.model;

import org.bytedeco.opencv.opencv_core.Mat;
import java.awt.Rectangle;
import java.util.LinkedList;

public class TrackedPerson {
    private Rectangle bounds;
    private Mat lastGrayRegion;
    private final LinkedList<Double> motionHistory = new LinkedList<>();
    private static final int HISTORY_SIZE = 60; // Approx 2-3 seconds at 20-30 fps
    private static final double MIN_SEIZURE_MOTION = 0.0; // Threshold to ignore sensor noise
    private boolean seizureDetected = false;
    private int lastSeenFrames = 0;
    private double cumulativeMotion = 0;

    public TrackedPerson(Rectangle bounds) {
        this.bounds = bounds;
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

    public void addMotion(double magnitude) {
        motionHistory.add(magnitude);
        cumulativeMotion += magnitude;
        if (motionHistory.size() > HISTORY_SIZE) {
            motionHistory.removeFirst();
        }
        analyzeSeizure();
    }

    private void analyzeSeizure() {
        if (motionHistory.size() < HISTORY_SIZE) {
            seizureDetected = false;
            return;
        }

        // Calculate mean motion magnitude
        double mean = motionHistory.stream().mapToDouble(d -> d).average().orElse(0.0);
        
        // BUG FIX #1: If motion is below the noise floor, it's not a seizure
        if (mean < MIN_SEIZURE_MOTION) {
            seizureDetected = false;
            return;
        }

        int peaks = 0;
        boolean above = false;
        
        for (Double val : motionHistory) {
            if (val > mean * 1.5 && !above) {
                peaks++;
                above = true;
            } else if (val < mean && above) {
                above = false;
            }
        }

        // Seizure rhythm check: 2Hz to 6Hz
        // If 60 frames = 2 seconds, then 4 to 12 peaks = 2-6Hz
        seizureDetected = (peaks >= 4 && peaks <= 15);
    }

    public boolean isSeizureDetected() { return seizureDetected; }
    
    // BUG FIX #2: If we've seen this object for many frames and it's barely moved,
    // it's probably furniture, not a person.
    public boolean isLikelyStaticObject() {
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
