package com.tapoviewer.model;

import org.bytedeco.opencv.opencv_core.Mat;
import java.awt.Rectangle;
import java.util.LinkedList;

public class TrackedPerson {
    private Rectangle bounds;
    private Mat lastGrayRegion;
    private final LinkedList<Double> motionHistory = new LinkedList<>();
    private static final int HISTORY_SIZE = 60; // Approx 2-3 seconds at 20-30 fps
    private boolean seizureDetected = false;
    private int lastSeenFrames = 0;

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

        // Calculate zero-crossing rate or peak frequency of the motion signal
        double mean = motionHistory.stream().mapToDouble(d -> d).average().orElse(0.0);
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
    
    public int incrementLastSeen() { return ++lastSeenFrames; }
    public void resetLastSeen() { lastSeenFrames = 0; }
    
    public void release() {
        if (lastGrayRegion != null) {
            lastGrayRegion.release();
            lastGrayRegion = null;
        }
    }
}
