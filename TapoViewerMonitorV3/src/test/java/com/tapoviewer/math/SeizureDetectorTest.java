package com.tapoviewer.math;

import com.tapoviewer.model.TrackedPerson;
import org.junit.jupiter.api.Test;

import java.awt.Rectangle;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the Spatio-Temporal Transformer integration:
 * <ul>
 *   <li>{@link SeizureDetector} ONNX model load and inference</li>
 *   <li>{@link TrackedPerson} skeletal feature buffer accumulation and normalisation</li>
 * </ul>
 *
 * <p>These tests require {@code seizure_transformer.onnx} to be present on the classpath
 * (i.e., in {@code src/main/resources/}).</p>
 */
public class SeizureDetectorTest {

    // ── Skeletal buffer tests (no GPU / ONNX required) ───────────────────────

    @Test
    public void testSkeletalBufferNotReadyWhenEmpty() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        assertFalse(person.isSkeletalBufferReady(),
                "Buffer should not be ready with 0 frames written.");
    }

    @Test
    public void testSkeletalBufferNotReadyPartiallyFilled() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        float[][] keypoints = buildDummyKeypoints(1.0f);
        for (int i = 0; i < SeizureDetector.SEQ_LEN - 1; i++) {
            person.addSkeletalFrame(keypoints, 1280, 720);
        }
        assertFalse(person.isSkeletalBufferReady(),
                "Buffer should not be ready with only " + (SeizureDetector.SEQ_LEN - 1) + " frames.");
    }

    @Test
    public void testSkeletalBufferReadyAfterFullWindow() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        float[][] keypoints = buildDummyKeypoints(0.9f);
        for (int i = 0; i < SeizureDetector.SEQ_LEN; i++) {
            person.addSkeletalFrame(keypoints, 1280, 720);
        }
        assertTrue(person.isSkeletalBufferReady(),
                "Buffer should be ready after exactly " + SeizureDetector.SEQ_LEN + " frames.");
    }

    @Test
    public void testSkeletalWindowDimensions() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        float[][] keypoints = buildDummyKeypoints(0.9f);
        for (int i = 0; i < SeizureDetector.SEQ_LEN; i++) {
            person.addSkeletalFrame(keypoints, 1280, 720);
        }
        float[][] window = person.getSkeletalWindow();
        assertNotNull(window, "Window must not be null when buffer is ready.");
        assertEquals(SeizureDetector.SEQ_LEN, window.length, "Window must have SEQ_LEN rows.");
        assertEquals(SeizureDetector.FEATURE_DIM, window[0].length, "Each row must have FEATURE_DIM columns.");
    }

    @Test
    public void testSkeletalNormalisationWithinBounds() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        // High-confidence keypoints at various pixel positions
        float[][] keypoints = buildDummyKeypoints(0.9f);
        for (int i = 0; i < SeizureDetector.SEQ_LEN; i++) {
            person.addSkeletalFrame(keypoints, 1280, 720);
        }
        float[][] window = person.getSkeletalWindow();
        for (float[] frame : window) {
            for (int j = 0; j < 17; j++) {
                float x    = frame[3 * j];
                float y    = frame[3 * j + 1];
                float conf = frame[3 * j + 2];
                assertTrue(x >= 0.0f && x <= 1.0f,
                        "Normalised x must be in [0,1], got: " + x);
                assertTrue(y >= 0.0f && y <= 1.0f,
                        "Normalised y must be in [0,1], got: " + y);
                // Confidence passthrough (raw from YOLO, not normalised)
                assertTrue(conf >= 0.0f && conf <= 1.0f,
                        "Confidence must be in [0,1], got: " + conf);
            }
        }
    }

    @Test
    public void testLowConfidenceJointsZeroedOut() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        // Build keypoints where all joints have confidence < 0.3
        float[][] keypoints = buildDummyKeypoints(0.1f); // low confidence
        for (int i = 0; i < SeizureDetector.SEQ_LEN; i++) {
            person.addSkeletalFrame(keypoints, 1280, 720);
        }
        float[][] window = person.getSkeletalWindow();
        for (float[] frame : window) {
            for (int j = 0; j < 17; j++) {
                assertEquals(0.0f, frame[3 * j],
                        "X should be zero for low-confidence joint " + j);
                assertEquals(0.0f, frame[3 * j + 1],
                        "Y should be zero for low-confidence joint " + j);
            }
        }
    }

    @Test
    public void testCircularBufferWrapsCorrectly() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        // Fill buffer beyond SEQ_LEN to confirm wrap-around doesn't crash
        float[][] keypoints = buildDummyKeypoints(0.9f);
        int extraFrames = SeizureDetector.SEQ_LEN * 3;
        for (int i = 0; i < extraFrames; i++) {
            person.addSkeletalFrame(keypoints, 1280, 720);
        }
        assertTrue(person.isSkeletalBufferReady(),
                "Buffer must remain ready after " + extraFrames + " frames.");
        float[][] window = person.getSkeletalWindow();
        assertNotNull(window, "Window must not be null after wrap-around.");
        assertEquals(SeizureDetector.SEQ_LEN, window.length, "Window length must equal SEQ_LEN after wrap.");
    }

    @Test
    public void testTransformerSeizureFlagDefaultFalse() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        assertFalse(person.isTransformerSeizureDetected(),
                "Transformer flag must be false by default.");
        assertEquals(0.0f, person.getTransformerSeizureProb(),
                "Transformer probability must be 0.0 by default.");
    }

    @Test
    public void testTransformerSeizureFlagSetterGetter() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        person.setTransformerSeizure(true);
        person.setTransformerSeizureProb(0.85f);
        assertTrue(person.isTransformerSeizureDetected());
        assertEquals(0.85f, person.getTransformerSeizureProb(), 0.001f);
    }

    @Test
    public void testIsSeizureConfirmedReflectsTransformerVerdictAlone() {
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        // No FFT motion added — only transformer verdict set
        person.setTransformerSeizure(true);
        assertTrue(person.isSeizureConfirmed(),
                "isSeizureConfirmed() must be true when only transformer says seizure.");
        assertTrue(person.isSeizureWarning(),
                "isSeizureWarning() must be true when only transformer says seizure.");
    }

    // ── ONNX inference tests (requires model on classpath) ───────────────────

    @Test
    public void testSeizureDetectorLoadsWithoutException() {
        // This will throw if seizure_transformer.onnx is not on the classpath
        assertDoesNotThrow(SeizureDetector::new,
                "SeizureDetector should initialise without throwing.");
    }

    @Test
    public void testInferenceProbabilityInRange() {
        try (SeizureDetector detector = new SeizureDetector()) {
            float[][] window = buildZeroWindow();
            float prob = detector.predictProbability(window);
            // Probability must be a valid softmax output in [0, 1] or -1 on error
            assertTrue(prob >= 0.0f && prob <= 1.0f,
                    "P(seizure) must be in [0, 1], got: " + prob);
        }
    }

    @Test
    public void testInferenceOnAllZeroWindowReturnsProbabilityInRange() {
        try (SeizureDetector detector = new SeizureDetector()) {
            // An all-zero window (all joints occluded) is an unusual input;
            // the model may classify it either way depending on training data.
            // The meaningful contract is that the output is a valid probability.
            float[][] window = buildZeroWindow();
            float prob = detector.predictProbability(window);
            assertTrue(prob >= 0.0f && prob <= 1.0f,
                    "P(seizure) for all-zero window must be a valid probability in [0,1], got: " + prob);
        }
    }

    @Test
    public void testInferenceOnNullWindowHandledGracefully() {
        try (SeizureDetector detector = new SeizureDetector()) {
            boolean result = detector.predict(null);
            assertFalse(result, "predict(null) should return false without throwing.");
        }
    }

    @Test
    public void testInferenceOnWrongLengthWindowHandledGracefully() {
        try (SeizureDetector detector = new SeizureDetector()) {
            float[][] badWindow = new float[10][SeizureDetector.FEATURE_DIM];
            boolean result = detector.predict(badWindow);
            assertFalse(result, "predict(wrong-length) should return false without throwing.");
        }
    }

    @Test
    public void testPredictProbabilityReturnsMinusOneOnBadInput() {
        try (SeizureDetector detector = new SeizureDetector()) {
            assertEquals(-1.0f, detector.predictProbability(null), 0.001f,
                    "predictProbability(null) should return -1.");
            assertEquals(-1.0f, detector.predictProbability(new float[5][SeizureDetector.FEATURE_DIM]), 0.001f,
                    "predictProbability(wrong length) should return -1.");
        }
    }

    @Test
    public void testConcurrentInferenceIsThreadSafe() throws InterruptedException {
        // Verifies that a single shared SeizureDetector can be called from multiple
        // threads simultaneously without crashing or returning nonsense probabilities.
        // This mirrors the 8-camera production setup where VideoPanel worker threads
        // all share one SeizureDetector instance.
        final int NUM_THREADS = 8;
        try (SeizureDetector detector = new SeizureDetector()) {
            Thread[] threads = new Thread[NUM_THREADS];
            float[] results = new float[NUM_THREADS];
            Throwable[] errors = new Throwable[NUM_THREADS];

            for (int t = 0; t < NUM_THREADS; t++) {
                final int idx = t;
                threads[t] = new Thread(() -> {
                    try {
                        float[][] window = buildZeroWindow();
                        results[idx] = detector.predictProbability(window);
                    } catch (Throwable e) {
                        errors[idx] = e;
                    }
                });
            }
            for (Thread th : threads) th.start();
            for (Thread th : threads) th.join(5000);

            for (int t = 0; t < NUM_THREADS; t++) {
                assertNull(errors[t], "Thread " + t + " threw: " + errors[t]);
                assertTrue(results[t] >= 0.0f && results[t] <= 1.0f,
                        "Thread " + t + " returned invalid probability: " + results[t]);
            }
        }
    }

    @Test
    public void testIsSeizureDetectedMatchesIsSeizureConfirmed() {
        // isSeizureDetected() and isSeizureConfirmed() should behave identically —
        // both include FFT confirmed state OR transformer verdict.
        TrackedPerson person = new TrackedPerson(new Rectangle(0, 0, 200, 400));
        // No state set: both should be false
        assertFalse(person.isSeizureDetected());
        assertFalse(person.isSeizureConfirmed());
        // Transformer only: both should be true
        person.setTransformerSeizure(true);
        assertTrue(person.isSeizureDetected());
        assertTrue(person.isSeizureConfirmed());
    }


    // ── Helpers ──────────────────────────────────────────────────────────────

    /**
     * Builds a {@code [17][3]} keypoints array with all joints at the centre of
     * a 1280×720 frame at the given confidence value.
     */
    private static float[][] buildDummyKeypoints(float confidence) {
        float[][] kpts = new float[17][3];
        for (int i = 0; i < 17; i++) {
            kpts[i][0] = 640.0f;  // x = frame centre
            kpts[i][1] = 360.0f;  // y = frame centre
            kpts[i][2] = confidence;
        }
        return kpts;
    }

    /** Returns an all-zero {@code [SEQ_LEN][FEATURE_DIM]} window. */
    private static float[][] buildZeroWindow() {
        return new float[SeizureDetector.SEQ_LEN][SeizureDetector.FEATURE_DIM];
    }
}

