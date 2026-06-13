package com.tapoviewer.cli;

import com.tapoviewer.math.CudaBridge;
import com.tapoviewer.math.YoloPoseDetector;
import com.tapoviewer.math.YoloPoseDetector.PoseDetection;
import com.tapoviewer.model.TrackedPerson;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;

import java.io.File;
import java.nio.ByteBuffer;
import java.lang.foreign.MemorySegment;
import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

public class VideoTester {
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java com.tapoviewer.cli.VideoTester <video_file_path>");
            return;
        }

        String videoPath = args[0];
        File file = new File(videoPath);
        if (!file.exists()) {
            System.err.println("File not found: " + videoPath);
            return;
        }

        System.out.println("Processing video: " + file.getAbsolutePath());
        
        // Suppress FFmpeg output noise
        org.bytedeco.ffmpeg.global.avutil.av_log_set_level(org.bytedeco.ffmpeg.global.avutil.AV_LOG_ERROR);

        try (YoloPoseDetector detector = new YoloPoseDetector();
             FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(file)) {
            
            grabber.start();
            System.out.println("Video started. Resolution: " + grabber.getImageWidth() + "x" + grabber.getImageHeight());
            System.out.println("Total frames: " + grabber.getLengthInVideoFrames());
            
            OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
            List<TrackedPerson> trackedPeople = new ArrayList<>();
            
            int frameIdx = 0;
            int seizureCount = 0;
            int framesWithPerson = 0;
            
            while (true) {
                Frame frame = grabber.grabImage();
                if (frame == null) break;
                frameIdx++;

                if (frameIdx > 1 && frameIdx % 2 == 0) {
                    continue;
                }
                
                Mat mat = toMatConverter.convert(frame);
                if (mat == null || mat.empty()) continue;
                
                // 1. Run YOLOv8-pose detection
                List<PoseDetection> detections = detector.detect(mat, 0.45f, 0.45f);
                if (!detections.isEmpty()) {
                    framesWithPerson++;
                }
                
                // 2. Track people (IoU based)
                List<TrackedPerson> matchedPeople = new ArrayList<>();
                for (PoseDetection det : detections) {
                    TrackedPerson bestMatch = null;
                    double maxIoU = 0.3;
                    
                    for (TrackedPerson p : trackedPeople) {
                        double iou = calculateIoU(det.bounds, p.getBounds());
                        if (iou > maxIoU) {
                            maxIoU = iou;
                            bestMatch = p;
                        }
                    }
                    
                    if (bestMatch != null) {
                        bestMatch.setBounds(det.bounds);
                        bestMatch.resetLastSeen();
                        trackedPeople.remove(bestMatch);
                        matchedPeople.add(bestMatch);
                    } else {
                        bestMatch = new TrackedPerson(det.bounds, true);
                        matchedPeople.add(bestMatch);
                    }
                    bestMatch.setLastKeypoints(det.keypoints);
                }
                
                // Cleanup old tracked people
                for (TrackedPerson p : trackedPeople) {
                    if (p.incrementLastSeen() < 5) {
                        matchedPeople.add(p);
                    } else {
                        p.release();
                    }
                }
                trackedPeople.clear();
                trackedPeople.addAll(matchedPeople);
                
                // 3. Update motion using joint vectors or Fallback FFM CUDA Optical Flow
                for (TrackedPerson p : trackedPeople) {
                    double motion = 0.0;
                    boolean poseValid = false;
                    
                    if (p.getLastKeypoints() != null && p.getPrevKeypoints() != null) {
                        int[] jointIndices = {9, 10, 15, 16}; // wrists & ankles
                        float[][] currKpts = p.getLastKeypoints();
                        float[][] prevKpts = p.getPrevKeypoints();
                        double maxDisplacement = 0.0;
                        int validJoints = 0;
                        
                        for (int idx : jointIndices) {
                            float[] currPt = currKpts[idx];
                            float[] prevPt = prevKpts[idx];
                            if (currPt[2] > 0.4f && prevPt[2] > 0.4f) {
                                double dist = Math.hypot(currPt[0] - prevPt[0], currPt[1] - prevPt[1]);
                                if (dist > maxDisplacement) {
                                    maxDisplacement = dist;
                                }
                                validJoints++;
                            }
                        }
                        if (validJoints > 0) {
                            motion = maxDisplacement;
                            poseValid = true;
                        }
                    }
                    
                    if (!poseValid) {
                        // CUDA Optical Flow fallback
                        Rectangle r = p.getBounds();
                        int x = Math.max(0, r.x);
                        int y = Math.max(0, r.y);
                        int w = Math.min(r.width, mat.cols() - x);
                        int h = Math.min(r.height, mat.rows() - y);
                        
                        if (w > 0 && h > 0) {
                            Mat region = new Mat(mat, new Rect(x, y, w, h));
                            Mat grayRegion = new Mat();
                            cvtColor(region, grayRegion, COLOR_BGR2GRAY);
                            
                            Mat lastGray = p.getLastGrayRegion();
                            if (lastGray != null && lastGray.rows() == grayRegion.rows() && lastGray.cols() == grayRegion.cols()) {
                                ByteBuffer currBuf = grayRegion.data().asByteBuffer();
                                ByteBuffer prevBuf = lastGray.data().asByteBuffer();
                                MemorySegment currSeg = MemorySegment.ofBuffer(currBuf);
                                MemorySegment prevSeg = MemorySegment.ofBuffer(prevBuf);
                                
                                float cudaMotion = CudaBridge.calculateMotionMagnitude(prevSeg, currSeg, w, h);
                                motion = cudaMotion * 15.0;
                            }
                            p.setLastGrayRegion(grayRegion);
                            region.release();
                        }
                    }
                    
                    p.addMotion(motion);
                    
                    if (p.isSeizureDetected()) {
                        seizureCount++;
                        System.out.printf("[Frame %d] Seizure detected! Peak Frequency: %.2f Hz\n", frameIdx, p.getPeakFrequencyHz());
                    }
                }
            }
            
            System.out.println("Processing complete. Total frames processed: " + frameIdx);
            System.out.println("Frames with person detected: " + framesWithPerson + " / " + frameIdx + " (" + String.format("%.1f", 100.0 * framesWithPerson / frameIdx) + "%)");
            System.out.println("Total seizure event frames flagged: " + seizureCount);
            
            for (TrackedPerson p : trackedPeople) p.release();
            
        } catch (Exception e) {
            System.err.println("Error processing video: " + e.getMessage());
            e.printStackTrace();
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
