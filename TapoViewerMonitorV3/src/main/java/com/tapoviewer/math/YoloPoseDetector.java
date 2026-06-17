package com.tapoviewer.math;

import ai.onnxruntime.*;
import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.Rectangle;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;

/**
 * YOLOv8-pose detector running via ONNX Runtime with the CUDA Execution Provider.
 *
 * <p><b>Priority 3 — GPU preprocessing:</b> The OpenCV CPU cvtColor + resize and
 * the Java CHW pixel loop have been replaced by {@link CudaBridge#preprocessFrameForYolo},
 * which runs the full pipeline on the GPU via NPP:</p>
 * <ol>
 *   <li>nppiResize_8u_C3R   — resize source frame to 640×640</li>
 *   <li>nppiSwapChannels_8u_C3R — BGR → RGB</li>
 *   <li>nppiConvert_8u32f_C3R  — uint8 → float32</li>
 *   <li>nppiDivC_32f_C3IR      — /255 normalise to [0,1]</li>
 *   <li>hwc_to_chw_kernel       — HWC → CHW rearrangement</li>
 * </ol>
 * <p>The resulting CHW float tensor is written into a pre-allocated direct
 * {@link ByteBuffer}, eliminating the ~5 ms Java loop on every frame.</p>
 */
public class YoloPoseDetector implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    /**
     * Pre-allocated direct ByteBuffer for the CHW float tensor.
     * Backed by off-heap native memory; safe to pass into the native preprocessing call.
     */
    private static class ThreadLocalBuffer {
        final ByteBuffer raw;
        final FloatBuffer floats;
        final MemorySegment segment;

        ThreadLocalBuffer() {
            this.raw = ByteBuffer.allocateDirect(3 * 640 * 640 * 4).order(ByteOrder.nativeOrder());
            this.floats = raw.asFloatBuffer();
            this.segment = MemorySegment.ofBuffer(raw);
        }
    }

    private final ThreadLocal<ThreadLocalBuffer> threadLocalBuffer = ThreadLocal.withInitial(ThreadLocalBuffer::new);

    public static class PoseDetection {
        public Rectangle bounds;
        public float confidence;
        public float[][] keypoints; // [17][3] -> [x, y, conf]

        public PoseDetection(Rectangle bounds, float confidence, float[][] keypoints) {
            this.bounds = bounds;
            this.confidence = confidence;
            this.keypoints = keypoints;
        }
    }

    public YoloPoseDetector() {
        try {
            this.env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            // Attempt to enable CUDA Execution Provider
            try {
                opts.addCUDA(0);
                System.out.println("YoloPoseDetector: CUDA Execution Provider enabled successfully.");
            } catch (Exception e) {
                System.err.println("YoloPoseDetector: CUDA EP not available, falling back to CPU: " + e.getMessage());
            }

            // Load model from resources
            try (InputStream is = YoloPoseDetector.class.getResourceAsStream("/yolov8n-pose.onnx")) {
                if (is == null) {
                    throw new RuntimeException("Could not find yolov8n-pose.onnx in resources.");
                }
                byte[] modelBytes = is.readAllBytes();
                this.session = env.createSession(modelBytes, opts);
            }

            this.inputName = session.getInputNames().iterator().next();


        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize YoloPoseDetector", e);
        }
    }

    public List<PoseDetection> detect(Mat mat, float confThreshold, float iouThreshold) {
        int origW = mat.cols();
        int origH = mat.rows();

        // ── Priority 3: GPU preprocessing (replaces CPU cvtColor + resize + Java loop) ──
        // Get the native pointer to the OpenCV Mat's BGR pixel data.
        long bgrAddr = mat.data().address();
        int step = (int) mat.step();
        MemorySegment bgrSeg = MemorySegment.ofAddress(bgrAddr)
                .reinterpret((long) origH * step);

        ThreadLocalBuffer buf = threadLocalBuffer.get();

        // Run the full GPU pipeline: resize → BGR→RGB → uint8→float32 → /255 → HWC→CHW.
        // Writes 3*640*640 floats into buf.segment (backed by buf.raw).
        CudaBridge.preprocessFrameForYolo(bgrSeg, origW, origH, step, buf.segment);

        // Rewind the FloatBuffer so ONNX reads from position 0
        buf.floats.rewind();

        List<PoseDetection> detections = new ArrayList<>();
        try {
            // Create ONNX tensor from the pre-filled direct FloatBuffer
            OnnxTensor inputTensor = OnnxTensor.createTensor(
                env, buf.floats, new long[]{1, 3, 640, 640});

            OrtSession.Result results;
            synchronized (session) {
                results = session.run(Collections.singletonMap(inputName, inputTensor));
            }
            try (results) {
                OnnxValue outputVal = results.get(0);
                float[][][] output = (float[][][]) outputVal.getValue(); // [1][56][8400]
                float[][] data = output[0]; // [56][8400]

                // Postprocess: filter by confidence, map back to original frame coordinates
                int numBoxes = data[0].length; // 8400
                for (int col = 0; col < numBoxes; col++) {
                    float conf = data[4][col];
                    if (conf > confThreshold) {
                        float cx = data[0][col];
                        float cy = data[1][col];
                        float w  = data[2][col];
                        float h  = data[3][col];

                        float scaleX = (float) origW / 640.0f;
                        float scaleY = (float) origH / 640.0f;

                        int xMin = (int) ((cx - w / 2.0f) * scaleX);
                        int yMin = (int) ((cy - h / 2.0f) * scaleY);
                        int boxW = (int) (w * scaleX);
                        int boxH = (int) (h * scaleY);

                        Rectangle bounds = new Rectangle(xMin, yMin, boxW, boxH);

                        // Extract 17 keypoints
                        float[][] kpts = new float[17][3];
                        for (int k = 0; k < 17; k++) {
                            float kx    = data[5 + k * 3][col] * scaleX;
                            float ky    = data[6 + k * 3][col] * scaleY;
                            float kconf = data[7 + k * 3][col];
                            kpts[k] = new float[]{kx, ky, kconf};
                        }

                        detections.add(new PoseDetection(bounds, conf, kpts));
                    }
                }
            }
            inputTensor.close();
        } catch (Exception e) {
            System.err.println("YoloPoseDetector: Error during inference: " + e.getMessage());
        }

        return nms(detections, iouThreshold);
    }

    private List<PoseDetection> nms(List<PoseDetection> detections, float iouThreshold) {
        detections.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<PoseDetection> result = new ArrayList<>();
        boolean[] active = new boolean[detections.size()];
        Arrays.fill(active, true);

        for (int i = 0; i < detections.size(); i++) {
            if (!active[i]) continue;
            PoseDetection a = detections.get(i);
            result.add(a);

            for (int j = i + 1; j < detections.size(); j++) {
                if (!active[j]) continue;
                PoseDetection b = detections.get(j);
                if (calculateIoU(a.bounds, b.bounds) > iouThreshold) {
                    active[j] = false;
                }
            }
        }
        return result;
    }

    private double calculateIoU(Rectangle r1, Rectangle r2) {
        int ix = Math.max(r1.x, r2.x);
        int iy = Math.max(r1.y, r2.y);
        int iw = Math.min(r1.x + r1.width,  r2.x + r2.width)  - ix;
        int ih = Math.min(r1.y + r1.height, r2.y + r2.height) - iy;

        if (iw <= 0 || ih <= 0) return 0;

        double intersectionArea = (double) iw * ih;
        double unionArea = (double)(r1.width * r1.height)
                         + (double)(r2.width * r2.height)
                         - intersectionArea;
        return intersectionArea / unionArea;
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
            threadLocalBuffer.remove();
        } catch (Exception e) {
            // Ignored
        }
    }
}
