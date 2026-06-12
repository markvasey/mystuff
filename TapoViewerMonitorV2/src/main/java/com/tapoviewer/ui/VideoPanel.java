package com.tapoviewer.ui;

import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import com.tapoviewer.model.PersonSnapshot;
import com.tapoviewer.model.TrackedPerson;
import com.tapoviewer.math.CudaBridge;
import com.tapoviewer.math.YoloPoseDetector;
import com.tapoviewer.math.YoloPoseDetector.PoseDetection;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

public class VideoPanel extends JPanel {
    private static final Logger logger = LoggerFactory.getLogger(VideoPanel.class);
    private FFmpegFrameGrabber grabber;
    private volatile boolean running = false;
    private BufferedImage currentFrame;
    private Thread workerThread;

    private final YoloPoseDetector yoloDetector;
    private final OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private final List<TrackedPerson> trackedPeople = new ArrayList<>();
    private final List<TrackedPerson> renderList = new CopyOnWriteArrayList<>();
    private Consumer<PersonSnapshot> snapshotListener;
    private LocalDateTime lastSnapshotTime = LocalDateTime.MIN;

    public VideoPanel() {
        setBackground(Color.BLACK);
        setLayout(new BorderLayout());

        // Initialize GPU-accelerated YOLO Pose Detector
        yoloDetector = new YoloPoseDetector();
    }

    public void setSnapshotListener(Consumer<PersonSnapshot> listener) {
        this.snapshotListener = listener;
    }

    public void play(String ip, int port, String stream, String username, String password) {
        play(ip, port, stream, username, password, false);
    }

    public void play(String ip, int port, String stream, String username, String password, boolean useGpuDecode) {
        stop(); // Ensure any previous stream is stopped

        // Suppress FFmpeg noise
        org.bytedeco.ffmpeg.global.avutil.av_log_set_level(org.bytedeco.ffmpeg.global.avutil.AV_LOG_ERROR);

        String mrl = String.format("rtsp://%s:%s@%s:%d/%s", username, password, ip, port, stream);
        logger.info("Connecting to RTSP via JavaCV: rtsp://{}:****@{}:{}/{}", username, ip, port, stream);

        running = true;
        workerThread = new Thread(() -> {
            try {
                grabber = new FFmpegFrameGrabber(mrl);
                grabber.setOption("rtsp_transport", "tcp"); // Force TCP for stability
                grabber.setOption("stimeout", "3000000");   // 3 second timeout
                if (useGpuDecode) {
                    grabber.setVideoCodecName("h264_cuvid");
                    logger.info("Enabling NVIDIA GPU Hardware Video Decoding (NVDEC)");
                }
                grabber.setVideoBitrate(0);                 // Auto
                grabber.start();

                Java2DFrameConverter converter = new Java2DFrameConverter();
                while (running) {
                    Frame frame = grabber.grabImage();
                    if (frame == null) break;

                    Mat mat = toMatConverter.convert(frame);
                    if (mat != null && !mat.empty()) {
                        // Run GPU YOLOv8-pose detection (conf: 0.5, iou: 0.45)
                        List<PoseDetection> detections = yoloDetector.detect(mat, 0.5f, 0.45f);
                        
                        updateTracking(mat, detections);
                    }

                    BufferedImage img = converter.getBufferedImage(frame);
                    if (img != null) {
                        currentFrame = img;
                        
                        // Handle Snapshots
                        if (snapshotListener != null && !renderList.isEmpty()) {
                            LocalDateTime now = LocalDateTime.now();
                            if (now.isAfter(lastSnapshotTime.plusSeconds(1))) {
                                for (TrackedPerson person : renderList) {
                                    Rectangle rect = person.getBounds();
                                    int x = Math.max(0, rect.x);
                                    int y = Math.max(0, rect.y);
                                    int w = Math.min(rect.width, img.getWidth() - x);
                                    int h = Math.min(rect.height, img.getHeight() - y);
                                    
                                    if (w > 0 && h > 0) {
                                        BufferedImage crop = img.getSubimage(x, y, w, h);
                                        BufferedImage copy = new BufferedImage(w, h, crop.getType());
                                        Graphics2D g2 = copy.createGraphics();
                                        g2.drawImage(crop, 0, 0, null);
                                        g2.dispose();
                                        
                                        snapshotListener.accept(new PersonSnapshot(copy, now));
                                        lastSnapshotTime = now;
                                        break;
                                    }
                                }
                            }
                        }
                        repaint();
                    }
                }
            } catch (Exception e) {
                logger.error("JavaCV Error: {}", e.getMessage());
            } finally {
                cleanup();
            }
        });
        workerThread.setDaemon(true);
        workerThread.start();
    }

    private void updateTracking(Mat fullMat, List<PoseDetection> detections) {
        List<TrackedPerson> matchedPeople = new ArrayList<>();
        
        for (PoseDetection det : detections) {
            TrackedPerson bestMatch = null;
            double maxIoU = 0.3; // Minimum overlap threshold
            
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
                bestMatch = new TrackedPerson(det.bounds);
                matchedPeople.add(bestMatch);
            }
            
            bestMatch.setLastKeypoints(det.keypoints);
        }
        
        // Cleanup old people
        for (TrackedPerson p : trackedPeople) {
            if (p.incrementLastSeen() < 5) {
                matchedPeople.add(p);
            } else {
                p.release();
            }
        }
        
        trackedPeople.clear();
        trackedPeople.addAll(matchedPeople);
        
        // Calculate motion & update signal history
        for (TrackedPerson p : trackedPeople) {
            double motion = 0.0;
            boolean poseValid = false;
            
            // 1. Joint Tracking (extremity velocity)
            if (p.getLastKeypoints() != null && p.getPrevKeypoints() != null) {
                int[] extremityIndices = {9, 10, 15, 16}; // Wrists and ankles
                float[][] currKpts = p.getLastKeypoints();
                float[][] prevKpts = p.getPrevKeypoints();
                
                double maxDisplacement = 0.0;
                int validJointsCount = 0;
                
                for (int idx : extremityIndices) {
                    float[] currPt = currKpts[idx];
                    float[] prevPt = prevKpts[idx];
                    
                    if (currPt[2] > 0.45f && prevPt[2] > 0.45f) {
                        double dist = Math.hypot(currPt[0] - prevPt[0], currPt[1] - prevPt[1]);
                        if (dist > maxDisplacement) {
                            maxDisplacement = dist;
                        }
                        validJointsCount++;
                    }
                }
                
                if (validJointsCount > 0) {
                    // Filter out keypoint jitter below 3.5 pixels
                    motion = (maxDisplacement < 3.5) ? 0.0 : maxDisplacement;
                    poseValid = true;
                }
            }
            
            // 2. Fallback: CUDA-accelerated motion quantification (under occlusion/blankets)
            if (!poseValid) {
                Rectangle r = p.getBounds();
                int x = Math.max(0, r.x);
                int y = Math.max(0, r.y);
                int w = Math.min(r.width, fullMat.cols() - x);
                int h = Math.min(r.height, fullMat.rows() - y);
                
                if (w > 0 && h > 0) {
                    Mat region = new Mat(fullMat, new Rect(x, y, w, h));
                    Mat grayRegion = new Mat();
                    cvtColor(region, grayRegion, COLOR_BGR2GRAY);
                    
                    Mat lastGray = p.getLastGrayRegion();
                    if (lastGray != null && lastGray.rows() == grayRegion.rows() && lastGray.cols() == grayRegion.cols()) {
                        ByteBuffer currBuf = grayRegion.data().asByteBuffer();
                        ByteBuffer prevBuf = lastGray.data().asByteBuffer();
                        
                        MemorySegment currSeg = MemorySegment.ofBuffer(currBuf);
                        MemorySegment prevSeg = MemorySegment.ofBuffer(prevBuf);
                        
                        // Execute FFM CUDA absolute differences kernel on GPU
                        float cudaMotion = CudaBridge.calculateMotionMagnitude(prevSeg, currSeg, w, h);
                        
                        // Map CUDA motion magnitude to displacement scale (roughly 15x multiplier)
                        double scaledCuda = cudaMotion * 15.0;
                        // Filter out camera noise/light flicker below 3.5 units
                        motion = (scaledCuda < 3.5) ? 0.0 : scaledCuda;
                    }
                    
                    p.setLastGrayRegion(grayRegion);
                    region.release();
                }
            }
            
            p.addMotion(motion);
        }
        
        renderList.clear();
        for (TrackedPerson p : trackedPeople) {
            if (!p.isLikelyStaticObject()) {
                renderList.add(p);
            }
        }
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
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (currentFrame != null) {
            int panelWidth = getWidth();
            int panelHeight = getHeight();
            int imgWidth = currentFrame.getWidth();
            int imgHeight = currentFrame.getHeight();

            double ratio = Math.min((double) panelWidth / imgWidth, (double) panelHeight / imgHeight);
            int newWidth = (int) (imgWidth * ratio);
            int newHeight = (int) (imgHeight * ratio);
            int x = (panelWidth - newWidth) / 2;
            int y = (panelHeight - newHeight) / 2;

            g.drawImage(currentFrame, x, y, newWidth, newHeight, null);

            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            
            for (TrackedPerson person : renderList) {
                Rectangle rect = person.getBounds();
                int rx = x + (int) (rect.x * ratio);
                int ry = y + (int) (rect.y * ratio);
                int rw = (int) (rect.width * ratio);
                int rh = (int) (rect.height * ratio);
                
                String stats = String.format("%.1f Hz (Amp: %.1f, Pow: %.1f)", 
                        person.getPeakFrequencyHz(), person.getPeakAmplitude(), person.getTotalPower());
                
                if (person.isSeizureDetected()) {
                    g2.setColor(Color.RED);
                    g2.setStroke(new BasicStroke(3.0f));
                    g2.drawRect(rx, ry, rw, rh);
                    g2.setFont(new Font("Arial", Font.BOLD, 16));
                    g2.drawString("SEIZURE DETECTED (" + stats + ")", rx, ry - 7);
                } else {
                    g2.setColor(Color.GREEN);
                    g2.setStroke(new BasicStroke(2.0f));
                    g2.drawRect(rx, ry, rw, rh);
                    g2.setFont(new Font("Arial", Font.PLAIN, 12));
                    g2.drawString(stats, rx, ry - 5);
                }

                // Render Skeleton Overlay
                if (person.getLastKeypoints() != null) {
                    drawSkeleton(g2, person.getLastKeypoints(), ratio, x, y);
                }
            }
        } else {
            g.setColor(Color.WHITE);
            g.drawString("No Video Signal", getWidth() / 2 - 40, getHeight() / 2);
        }
    }

    private void drawSkeleton(Graphics2D g2, float[][] keypoints, double ratio, int x, int y) {
        int[][] connections = {
            {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // Upper body
            {11, 12}, {11, 13}, {13, 15}, {12, 14}, {14, 16}, // Lower body
            {5, 11}, {6, 12} // Torso sides
        };

        // Draw connections
        g2.setColor(new Color(0, 255, 0, 180));
        g2.setStroke(new BasicStroke(2.0f));
        for (int[] conn : connections) {
            float[] p1 = keypoints[conn[0]];
            float[] p2 = keypoints[conn[1]];
            if (p1[2] > 0.45f && p2[2] > 0.45f) {
                int x1 = x + (int) (p1[0] * ratio);
                int y1 = y + (int) (p1[1] * ratio);
                int x2 = x + (int) (p1[0] * ratio);
                int y2 = y + (int) (p1[1] * ratio);
                
                // Let's connect p1 and p2 correctly
                x2 = x + (int) (p2[0] * ratio);
                y2 = y + (int) (p2[1] * ratio);
                g2.drawLine(x1, y1, x2, y2);
            }
        }

        // Draw joint points
        g2.setColor(Color.YELLOW);
        for (int i = 0; i < 17; i++) {
            float[] pt = keypoints[i];
            if (pt[2] > 0.45f) {
                int kx = x + (int) (pt[0] * ratio);
                int ky = y + (int) (pt[1] * ratio);
                g2.fillOval(kx - 3, ky - 3, 6, 6);
            }
        }
    }

    public void stop() {
        running = false;
        if (workerThread != null) {
            try {
                workerThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        cleanup();
    }

    private void cleanup() {
        try {
            if (grabber != null) {
                grabber.stop();
                grabber.release();
                grabber = null;
            }
        } catch (Exception e) {
            logger.warn("Cleanup error: {}", e.getMessage());
        }
        currentFrame = null;
        for (TrackedPerson p : trackedPeople) p.release();
        trackedPeople.clear();
        renderList.clear();
        repaint();
    }

    public void release() {
        stop();
        if (yoloDetector != null) {
            yoloDetector.close();
        }
    }
}
