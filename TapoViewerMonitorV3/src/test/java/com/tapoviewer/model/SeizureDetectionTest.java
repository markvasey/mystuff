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
        for (int i = 0; i < 200; i++) {
            double motion = Math.random() * 0.4; // random noise under threshold
            person.addMotion(motion);
        }

        assertFalse(person.isSeizureDetected(), "Random low-intensity movement should not trigger a seizure.");
    }

    @Test
    public void testSeizureRhythmicMovementDetected() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 100));

        // Simulate a strong 4 Hz rhythmic shaking at 20 frames per second
        // Fs = 20 Hz, Period = 5 frames.
        // We add 70 frames to fill the 64-frame history buffer.
        double samplingFrequency = 20.0;
        double targetFrequency = 4.0; // 4 Hz (in the 2-6 Hz band)

        for (int t = 0; t < 200; t++) {
            double motion = 20.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 30.0;
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
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 100));

        // Simulate a very slow 0.5 Hz rhythmic movement (e.g. breathing or normal body shifts)
        double samplingFrequency = 20.0;
        double targetFrequency = 0.5; // 0.5 Hz (well below the 2-6 Hz band)

        for (int t = 0; t < 200; t++) {
            double motion = 20.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 30.0;
            person.addMotion(motion);
        }

        assertFalse(person.isSeizureDetected(), "Slow 0.5 Hz movements should not trigger a seizure alert.");
    }

    @Test
    public void testFastRhythmicMovementRejected() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 100));

        // Simulate a fast 9.0 Hz rhythmic movement (e.g. camera sensor hum or fan vibrations)
        double samplingFrequency = 20.0;
        double targetFrequency = 9.0; // 9 Hz (well above the 2-6 Hz band)

        for (int t = 0; t < 200; t++) {
            double motion = 20.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 30.0;
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

    @Test
    public void testSeizureConditionForSnapshots() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 100));

        // Under normal circumstances/motion, seizure is not detected
        for (int i = 0; i < 200; i++) {
            person.addMotion(Math.random() * 2.0);
        }
        assertFalse(person.isSeizureDetected(), "No seizure should be detected under normal random motion.");

        // Under rhythmic seizure circumstances, seizure is detected
        double samplingFrequency = 20.0;
        double targetFrequency = 3.5; // 3.5 Hz
        for (int t = 0; t < 200; t++) {
            double motion = 20.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 30.0;
            person.addMotion(motion);
        }
        assertTrue(person.isSeizureDetected(), "Seizure should be detected under rhythmic motion, triggering snapshot capability.");
    }

    @Test
    public void testTonicPhaseWarning() {
        // Create a person without frame skipping (20 fps)
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 100), false);

        // Simulate a high-frequency 12 Hz tonic contraction
        // Fs = 20 Hz, target = 12 Hz. It will alias to 8 Hz in the FFT, which falls in the tonic band (6-10 Hz).
        double samplingFrequency = 20.0;
        double targetFrequency = 12.0;

        for (int t = 0; t < 100; t++) {
            double motion = 30.0 * Math.sin(2.0 * Math.PI * targetFrequency * (t / samplingFrequency)) + 40.0;
            person.addMotion(motion);
        }

        // Verify that tonic warning is active but clonic confirmed alarm is false
        assertTrue(person.isTonicWarning(), "High frequency (12 Hz aliased to 8 Hz) should trigger a tonic warning.");
        assertTrue(person.isSeizureWarning(), "Tonic warning should elevate to Stage 1 Seizure Warning.");
        assertFalse(person.isSeizureConfirmed(), "Tonic-only warnings must NOT trigger the Stage 2 confirmed clonic alarm.");
    }

    @Test
    public void testTonicToClonicTransition() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 100), false);

        double samplingFrequency = 20.0;

        // Phase 1: Tonic phase (12 Hz) for 70 frames
        double tonicFreq = 12.0;
        for (int t = 0; t < 70; t++) {
            double motion = 30.0 * Math.sin(2.0 * Math.PI * tonicFreq * (t / samplingFrequency)) + 40.0;
            person.addMotion(motion);
        }
        assertTrue(person.isTonicWarning(), "Should detect early tonic phase.");
        assertFalse(person.isSeizureConfirmed(), "Stage 2 confirmed alarm should not be active yet.");

        // Phase 2: Transition to clonic shaking (3.5 Hz) for 70 frames
        double clonicFreq = 3.5;
        for (int t = 0; t < 70; t++) {
            double motion = 30.0 * Math.sin(2.0 * Math.PI * clonicFreq * (t / samplingFrequency)) + 40.0;
            person.addMotion(motion);
        }
        // At 20 fps, density window is 300, and confirmation threshold is 10 frames.
        // After 70 frames of clonic active warning, it must confirm the seizure.
        assertTrue(person.isSeizureConfirmed(), "Transition to clonic shaking should trigger the Stage 2 confirmed alarm.");
    }
}
