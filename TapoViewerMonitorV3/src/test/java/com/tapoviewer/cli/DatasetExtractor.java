package com.tapoviewer.cli;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.tapoviewer.math.YoloPoseDetector;
import com.tapoviewer.math.YoloPoseDetector.PoseDetection;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;

import java.awt.Rectangle;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * CLI tool to recursively extract normalized skeletal coordinate sequences from training
 * and evaluation video directories, and save them as structured JSON files for PyTorch training.
 */
public class DatasetExtractor {

    public static class FrameFeatures {
        public int frameIdx;
        public float[] features; // 17 joints * 3 (x, y, conf) = 51 values

        public FrameFeatures() {}
        public FrameFeatures(int frameIdx, float[] features) {
            this.frameIdx = frameIdx;
            this.features = features;
        }
    }

    public static class VideoDataset {
        public String filename;
        public int label; // 1 for Seizure, 0 for Non-Seizure
        public int width;
        public int height;
        public double fps;
        public int totalFrames;
        public List<FrameFeatures> sequence = new ArrayList<>();
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java com.tapoviewer.cli.DatasetExtractor <dataset_dir> <label_value (0 or 1)>");
            System.out.println("Example: java com.tapoviewer.cli.DatasetExtractor TestVideos/Training_Calibration_Seizures 1");
            return;
        }

        String datasetDirPath = args[0];
        int label = Integer.parseInt(args[1]);

        File datasetDir = new File(datasetDirPath);
        if (!datasetDir.exists() || !datasetDir.isDirectory()) {
            System.err.println("Error: Directory not found: " + datasetDirPath);
            System.exit(1);
        }

        File[] videoFiles = datasetDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".mp4"));
        if (videoFiles == null || videoFiles.length == 0) {
            System.out.println("No MP4 files found in " + datasetDirPath);
            return;
        }

        System.out.println("Starting skeletal extraction on directory: " + datasetDir.getAbsolutePath());
        System.out.println("Found " + videoFiles.length + " video(s). Target Label: " + label);

        // Suppress FFmpeg verbose logging
        org.bytedeco.ffmpeg.global.avutil.av_log_set_level(org.bytedeco.ffmpeg.global.avutil.AV_LOG_ERROR);

        try (YoloPoseDetector detector = new YoloPoseDetector()) {
            ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

            for (File videoFile : videoFiles) {
                System.out.println("\nProcessing: " + videoFile.getName());
                long startTime = System.currentTimeMillis();

                VideoDataset dataset = extractVideoSkeletalSequence(videoFile, detector, label);

                if (dataset != null && !dataset.sequence.isEmpty()) {
                    File outputFile = new File(videoFile.getParent(), videoFile.getName().substring(0, videoFile.getName().lastIndexOf('.')) + ".json");
                    mapper.writeValue(outputFile, dataset);
                    long duration = System.currentTimeMillis() - startTime;
                    System.out.println("Saved extraction data to: " + outputFile.getName() + " (" + duration + " ms, " + dataset.sequence.size() + " frames extracted)");
                } else {
                    System.err.println("Warning: Extraction yielded empty sequence for: " + videoFile.getName());
                }
            }
            System.out.println("\nExtraction completed successfully!");

        } catch (Exception e) {
            System.err.println("Error running skeletal extraction: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static VideoDataset extractVideoSkeletalSequence(File videoFile, YoloPoseDetector detector, int label) {
        VideoDataset dataset = new VideoDataset();
        dataset.filename = videoFile.getName();
        dataset.label = label;

        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFile)) {
            grabber.start();
            dataset.width = grabber.getImageWidth();
            dataset.height = grabber.getImageHeight();
            dataset.fps = grabber.getFrameRate();
            dataset.totalFrames = grabber.getLengthInVideoFrames();

            float maxDim = Math.max(dataset.width, dataset.height);
            OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();

            int frameIdx = 0;
            // Retain last known keypoints to carry forward during occlusion/missing frames
            float[] lastKnownFeatures = new float[51]; // 17 joints * 3
            Rectangle lastBounds = null;

            while (true) {
                Frame frame = grabber.grabImage();
                if (frame == null) break;
                frameIdx++;

                // Skip even frames to match the 10 FPS runtime frame-skipping pipeline
                if (frameIdx > 1 && frameIdx % 2 == 0) {
                    continue;
                }

                Mat mat = toMatConverter.convert(frame);
                if (mat == null || mat.empty()) continue;

                List<PoseDetection> detections = detector.detect(mat, 0.40f, 0.40f);

                PoseDetection bestMatch = null;

                if (!detections.isEmpty()) {
                    if (lastBounds == null) {
                        // First frame with person: track largest bounding box
                        double maxArea = 0;
                        for (PoseDetection det : detections) {
                            double area = det.bounds.width * det.bounds.height;
                            if (area > maxArea) {
                                maxArea = area;
                                bestMatch = det;
                            }
                        }
                    } else {
                        // Track using IoU (Intersection-over-Union) to preserve subject identity
                        double maxIoU = 0;
                        for (PoseDetection det : detections) {
                            double iou = calculateIoU(det.bounds, lastBounds);
                            if (iou > 0.15 && iou > maxIoU) {
                                maxIoU = iou;
                                bestMatch = det;
                            }
                        }
                        
                        // Fallback: if tracking is lost, pick the largest detection
                        if (bestMatch == null) {
                            double maxArea = 0;
                            for (PoseDetection det : detections) {
                                double area = det.bounds.width * det.bounds.height;
                                if (area > maxArea) {
                                    maxArea = area;
                                    bestMatch = det;
                                }
                            }
                        }
                    }
                }

                float[] currentFeatures = new float[51];

                if (bestMatch != null) {
                    lastBounds = bestMatch.bounds;
                    float[][] keypoints = bestMatch.keypoints;

                    // Extract and normalize coordinates relative to maxDim
                    for (int i = 0; i < 17; i++) {
                        currentFeatures[3 * i]     = keypoints[i][0] / maxDim; // Normalized X
                        currentFeatures[3 * i + 1] = keypoints[i][1] / maxDim; // Normalized Y
                        currentFeatures[3 * i + 2] = keypoints[i][2];          // Raw Confidence
                    }
                    System.arraycopy(currentFeatures, 0, lastKnownFeatures, 0, 51);
                } else {
                    // If subject is occluded/missing, carry forward last known features with a small decay factor on confidence
                    System.arraycopy(lastKnownFeatures, 0, currentFeatures, 0, 51);
                    for (int i = 0; i < 17; i++) {
                        currentFeatures[3 * i + 2] *= 0.90f; // Decay confidence by 10% per skipped frame
                    }
                    System.arraycopy(currentFeatures, 0, lastKnownFeatures, 0, 51);
                }

                dataset.sequence.add(new FrameFeatures(frameIdx, currentFeatures));
            }

            grabber.stop();
            return dataset;

        } catch (Exception e) {
            System.err.println("Error extracting features from " + videoFile.getName() + ": " + e.getMessage());
            return null;
        }
    }

    private static double calculateIoU(Rectangle r1, Rectangle r2) {
        int intersectionX = Math.max(r1.x, r2.x);
        int intersectionY = Math.max(r1.y, r2.y);
        int intersectionW = Math.min(r1.x + r1.width, r2.x + r2.width) - intersectionX;
        int intersectionH = Math.min(r1.y + r1.height, r2.y + r2.height) - intersectionY;

        if (intersectionW <= 0 || intersectionH <= 0) return 0;

        double intersectionArea = intersectionW * intersectionH;
        double unionArea = (r1.width * r1.height) + (r2.width * r2.height) - intersectionArea;

        return intersectionArea / unionArea;
    }
}
