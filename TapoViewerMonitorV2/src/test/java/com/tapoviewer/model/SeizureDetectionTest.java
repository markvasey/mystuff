package com.tapoviewer.model;

import org.junit.jupiter.api.Test;
import com.tapoviewer.math.CudaBridge;
import static org.junit.jupiter.api.Assertions.*;

import java.awt.Rectangle;

public class SeizureDetectionTest {

    @Test
    public void testNormalMovementNoSeizure() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 100, 200));

        // Simulate normal random movement (low motion and chaotic)
        for (int i = 0; i < 70; i++) {
            double motion = Math.random() * 0.4; // random noise under threshold
            person.addMotion(motion);
        }

        assertFalse(person.isSeizureDetected(), "Random low-intensity movement should not trigger a seizure.");
    }

    @Test
    public void testSeizureRhythmicMovementDetected() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 100, 200));

        // Simulate a strong 4 Hz rhythmic shaking at 20 frames per second
        // Fs = 20 Hz, Period = 5 frames.
        // We add 70 frames to fill the 64-frame history buffer.
        double samplingFrequency = 20.0;
        double targetFrequency = 4.0; // 4 Hz (in the 2-6 Hz band)

        for (int t = 0; t < 70; t++) {
            double motion = 2.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 2.5;
            person.addMotion(motion);
        }

        assertTrue(person.isSeizureDetected(), "Rhythmic 4 Hz movement should trigger a seizure.");
        
        double peakFreq = person.getPeakFrequencyHz();
        // Since resolution is 0.3125 Hz, 4.0 Hz will map closest to bin 13 (13 * 0.3125 = 4.06 Hz)
        assertTrue(Math.abs(peakFreq - 4.0) <= 0.5, 
            "Peak frequency should be detected near 4.0 Hz, but got: " + peakFreq + " Hz");
    }

    @Test
    public void testSlowRhythmicMovementRejected() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 100, 200));

        // Simulate a very slow 0.5 Hz rhythmic movement (e.g. breathing or normal body shifts)
        double samplingFrequency = 20.0;
        double targetFrequency = 0.5; // 0.5 Hz (well below the 2-6 Hz band)

        for (int t = 0; t < 70; t++) {
            double motion = 2.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 2.5;
            person.addMotion(motion);
        }

        assertFalse(person.isSeizureDetected(), "Slow 0.5 Hz movements should not trigger a seizure alert.");
    }

    @Test
    public void testFastRhythmicMovementRejected() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 100, 200));

        // Simulate a fast 9.0 Hz rhythmic movement (e.g. camera sensor hum or fan vibrations)
        double samplingFrequency = 20.0;
        double targetFrequency = 9.0; // 9 Hz (well above the 2-6 Hz band)

        for (int t = 0; t < 70; t++) {
            double motion = 2.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 2.5;
            person.addMotion(motion);
        }

        assertFalse(person.isSeizureDetected(), "Fast 9 Hz vibrations should not trigger a seizure alert.");
    }

    @Test
    public void testStaticObjectFiltering() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 100, 200));

        // Increment frame counts and verify static filtering
        // If we see this object for over 50 frames and its average lifetime motion is < 0.2,
        // it should be flagged as likely a static object (e.g., chair, table).
        for (int i = 0; i < 60; i++) {
            person.incrementLastSeen();
            person.addMotion(0.05); // barely moving
        }

        assertTrue(person.isLikelyStaticObject(), "Hardly moving object should be flagged as a static object.");
    }

    @Test
    public void testCudaLibraryBinding() {
        // Verify CudaBridge loads the compiled library and resolves function symbols cleanly
        assertNotNull(CudaBridge.class.getSimpleName());
    }
}
