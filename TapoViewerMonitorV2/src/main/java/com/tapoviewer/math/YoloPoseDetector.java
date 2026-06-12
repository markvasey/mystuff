package com.tapoviewer.math;

import ai.onnxruntime.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2RGB;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

import java.awt.Rectangle;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.*;

public class YoloPoseDetector implements AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

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
                System.err.println("YoloPoseDetector: CUDA Execution Provider not available. Falling back to CPU: " + e.getMessage());
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
        
        // 1. Preprocess: Resize frame to 640x640
        Mat rgbMat = new Mat();
        cvtColor(mat, rgbMat, COLOR_BGR2RGB);
        
        Mat resizedMat = new Mat();
        resize(rgbMat, resizedMat, new Size(640, 640));
        
        // Convert to float planar format [1, 3, 640, 640]
        float[] inputData = new float[1 * 3 * 640 * 640];
        
        // Efficient pixel extraction using ByteBuffer
        ByteBuffer byteBuf = resizedMat.createBuffer();
        int rowStride = resizedMat.cols() * 3;
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                int baseIndex = y * rowStride + x * 3;
                float r = (byteBuf.get(baseIndex) & 0xFF) / 255.0f;
                float g = (byteBuf.get(baseIndex + 1) & 0xFF) / 255.0f;
                float b = (byteBuf.get(baseIndex + 2) & 0xFF) / 255.0f;
                
                inputData[0 * 640 * 640 + y * 640 + x] = r;
                inputData[1 * 640 * 640 + y * 640 + x] = g;
                inputData[2 * 640 * 640 + y * 640 + x] = b;
            }
        }
        
        resizedMat.release();
        rgbMat.release();

        List<PoseDetection> detections = new ArrayList<>();
        try {
            // Create ONNX Tensor
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), new long[]{1, 3, 640, 640});
            
            try (OrtSession.Result results = session.run(Collections.singletonMap(inputName, inputTensor))) {
                OnnxValue outputVal = results.get(0);
                float[][][] output = (float[][][]) outputVal.getValue(); // [1][56][8400]
                float[][] data = output[0]; // [56][8400]
                
                // Postprocess
                int numBoxes = data[0].length; // 8400
                for (int col = 0; col < numBoxes; col++) {
                    float conf = data[4][col];
                    if (conf > confThreshold) {
                        float cx = data[0][col];
                        float cy = data[1][col];
                        float w = data[2][col];
                        float h = data[3][col];
                        
                        // Map coordinates from 640x640 back to original frame size
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
                            float kx = data[5 + k * 3][col] * scaleX;
                            float ky = data[6 + k * 3][col] * scaleY;
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
        int intersectionX = Math.max(r1.x, r2.x);
        int intersectionY = Math.max(r1.y, r2.y);
        int intersectionW = Math.min(r1.x + r1.width, r2.x + r2.width) - intersectionX;
        int intersectionH = Math.min(r1.y + r1.height, r2.y + r2.height) - intersectionY;

        if (intersectionW <= 0 || intersectionH <= 0) return 0;

        double intersectionArea = intersectionW * intersectionH;
        double unionArea = (r1.width * r1.height) + (r2.width * r2.height) - intersectionArea;

        return intersectionArea / unionArea;
    }

    @Override
    public void close() {
        try {
            if (session != null) session.close();
            if (env != null) env.close();
        } catch (Exception e) {
            // Ignored
        }
    }
}
