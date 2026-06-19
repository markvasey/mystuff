package com.tapoviewer.math;

import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * Runs the trained {@code seizure_transformer.onnx} model via ONNX Runtime.
 *
 * <p>The model accepts a single tensor of shape {@code [1, SEQ_LEN, FEATURE_DIM]}
 * (float32) and outputs a softmax probability vector of shape {@code [1, 2]}:
 * <ul>
 *   <li>index 0 → P(no seizure)</li>
 *   <li>index 1 → P(seizure)</li>
 * </ul>
 *
 * <p>This class is <b>thread-safe</b>: inference is synchronised on the
 * {@link OrtSession} instance, matching the pattern used in {@link YoloPoseDetector}.
 */
public class SeizureDetector implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(SeizureDetector.class);

    /** Sequence-window length expected by the trained model. */
    public static final int SEQ_LEN = 32;

    /** Number of features per frame: 17 joints × 3 (x, y, conf). */
    public static final int FEATURE_DIM = 51;

    /** Minimum P(seizure) required to raise a transformer alarm. */
    private static final float SEIZURE_THRESHOLD = 0.65f;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    public SeizureDetector() {
        try {
            this.env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            // Prefer CUDA EP; fall back silently to CPU
            try {
                opts.addCUDA(0);
                logger.info("SeizureDetector: CUDA Execution Provider enabled.");
            } catch (Exception e) {
                logger.warn("SeizureDetector: CUDA EP not available, using CPU: {}", e.getMessage());
            }

            try (InputStream is = SeizureDetector.class.getResourceAsStream("/seizure_transformer.onnx")) {
                if (is == null) {
                    throw new RuntimeException(
                            "Could not find seizure_transformer.onnx in resources. " +
                            "Copy the trained model to src/main/resources/ and rebuild.");
                }
                byte[] modelBytes = is.readAllBytes();
                this.session = env.createSession(modelBytes, opts);
            }

            this.inputName = session.getInputNames().iterator().next();
            logger.info("SeizureDetector: model loaded. Input '{}', output names: {}",
                    inputName, session.getOutputNames());

        } catch (Exception e) {
            throw new RuntimeException("Failed to initialise SeizureDetector", e);
        }
    }

    /**
     * Runs inference on a {@code [SEQ_LEN × FEATURE_DIM]} feature window.
     *
     * <p>This method is thread-safe: a local {@link FloatBuffer} is created per call
     * (cheap — 32×51×4 = 6.5 KB stack-friendly allocation) so concurrent calls from
     * multiple camera threads cannot corrupt each other's input data.
     *
     * @param window row-major float array of shape {@code [SEQ_LEN][FEATURE_DIM]}.
     *               Each row is one frame's normalised skeletal feature vector.
     * @return {@code true} when the model predicts a seizure with sufficient confidence.
     */
    public boolean predict(float[][] window) {
        if (window == null || window.length != SEQ_LEN) {
            logger.warn("SeizureDetector.predict: unexpected window length {}",
                    window == null ? "null" : window.length);
            return false;
        }

        // Local buffer per call — safe for concurrent use across 8 camera threads.
        FloatBuffer buf = FloatBuffer.allocate(SEQ_LEN * FEATURE_DIM);
        for (float[] frame : window) {
            buf.put(frame, 0, FEATURE_DIM);
        }
        buf.rewind();

        try {
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    env, buf, new long[]{1, SEQ_LEN, FEATURE_DIM});

            OrtSession.Result result;
            synchronized (session) {
                result = session.run(Collections.singletonMap(inputName, inputTensor));
            }

            try (result) {
                float[][] probs = (float[][]) result.get(0).getValue(); // [1][2]
                float seizureProb = probs[0][1];
                logger.debug("SeizureDetector: P(seizure)={}", seizureProb);
                return seizureProb >= SEIZURE_THRESHOLD;
            } finally {
                inputTensor.close();
            }

        } catch (Exception e) {
            logger.error("SeizureDetector: inference error: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Returns the raw seizure probability for the given window, for diagnostic use.
     *
     * @param window row-major float array of shape {@code [SEQ_LEN][FEATURE_DIM]}.
     * @return P(seizure) in [0, 1], or {@code -1.0f} on error.
     */
    public float predictProbability(float[][] window) {
        if (window == null || window.length != SEQ_LEN) return -1.0f;

        // Local buffer per call — safe for concurrent use.
        FloatBuffer buf = FloatBuffer.allocate(SEQ_LEN * FEATURE_DIM);
        for (float[] frame : window) {
            buf.put(frame, 0, FEATURE_DIM);
        }
        buf.rewind();

        try {
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    env, buf, new long[]{1, SEQ_LEN, FEATURE_DIM});

            OrtSession.Result result;
            synchronized (session) {
                result = session.run(Collections.singletonMap(inputName, inputTensor));
            }

            try (result) {
                float[][] probs = (float[][]) result.get(0).getValue();
                return probs[0][1];
            } finally {
                inputTensor.close();
            }

        } catch (Exception e) {
            logger.error("SeizureDetector: probability inference error: {}", e.getMessage());
            return -1.0f;
        }
    }

    @Override
    public void close() {
        // Only close the session — OrtEnvironment is a process-wide singleton
        // and must not be closed while other sessions (e.g. YoloPoseDetector) are running.
        try {
            if (session != null) session.close();
        } catch (Exception ignored) {}
    }
}
