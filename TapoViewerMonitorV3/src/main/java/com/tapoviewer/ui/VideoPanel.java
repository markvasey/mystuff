package com.tapoviewer.ui;

import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import com.tapoviewer.model.PersonSnapshot;
import com.tapoviewer.model.TrackedPerson;
import com.tapoviewer.math.CudaBridge;
import com.tapoviewer.math.SeizureDetector;
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
    private final SeizureDetector seizureDetector;
    private final boolean ownsDetector;   // owns yoloDetector
    private final boolean ownsSeizureDetector; // owns seizureDetector
    private String cameraName = "Tapo Camera";
    private boolean selected = false;
    private String connectionError = null;

    private final OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private final List<TrackedPerson> trackedPeople = new ArrayList<>();
    private final List<TrackedPerson> renderList = new CopyOnWriteArrayList<>();
    private Consumer<PersonSnapshot> snapshotListener;
    private LocalDateTime lastSnapshotTime = LocalDateTime.MIN;

    public VideoPanel() {
        this.yoloDetector = new YoloPoseDetector();
        this.seizureDetector = new SeizureDetector();
        this.ownsDetector = true;
        this.ownsSeizureDetector = true;
        setBackground(Color.BLACK);
        setLayout(new BorderLayout());
    }

    public VideoPanel(YoloPoseDetector yoloDetector) {
        this.yoloDetector = yoloDetector;
        this.seizureDetector = new SeizureDetector();
        this.ownsDetector = false;
        this.ownsSeizureDetector = true;
        setBackground(Color.BLACK);
        setLayout(new BorderLayout());
    }

    /** Used by VideoGridPanel to share a single SeizureDetector across all feeds. */
    public VideoPanel(YoloPoseDetector yoloDetector, SeizureDetector seizureDetector) {
        this.yoloDetector = yoloDetector;
        this.seizureDetector = seizureDetector;
        this.ownsDetector = false;
        this.ownsSeizureDetector = false;
        setBackground(Color.BLACK);
        setLayout(new BorderLayout());
    }

    public String getCameraName() {
        return cameraName;
    }

    public void setSelected(boolean selected) {
        this.selected = selected;
        repaint();
    }

    public void setSnapshotListener(Consumer<PersonSnapshot> listener) {
        this.snapshotListener = listener;
    }

    public void play(String cameraName, String ip, int port, String stream, String username, String password, boolean useGpuDecode) {
        this.cameraName = cameraName;
        play(ip, port, stream, username, password, useGpuDecode);
    }

    public void play(String ip, int port, String stream, String username, String password) {
        play(ip, port, stream, username, password, false);
    }

    public void play(String ip, int port, String stream, String username, String password, boolean useGpuDecode) {
        stop(); // Ensure any previous stream is stopped
        this.connectionError = "Connecting...";

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
                int frameIdx = 0;
                while (running) {
                    Frame frame = grabber.grabImage();
                    if (frame == null) break;
                    connectionError = null; // Successful connection!

                    Mat mat = toMatConverter.convert(frame);
                    if (mat != null && !mat.empty()) {
                        frameIdx++;
                        if (frameIdx > 1 && frameIdx % 2 == 0) {
                            // Skip processing entirely on even frames.
                            // The renderList and trackedPeople remain unchanged, preserving smooth UI rendering.
                        } else {
                            // Run GPU YOLOv8-pose detection (conf: 0.5, iou: 0.45)
                            List<PoseDetection> detections = yoloDetector.detect(mat, 0.5f, 0.45f);
                            updateTracking(mat, detections);
                        }
                    }

                    BufferedImage img = converter.getBufferedImage(frame);
                    if (img != null) {
                        currentFrame = img;
                        
                        // Handle Snapshots
                        if (snapshotListener != null && !renderList.isEmpty()) {
                            LocalDateTime now = LocalDateTime.now();
                            if (now.isAfter(lastSnapshotTime.plusSeconds(1))) {
                                for (TrackedPerson person : renderList) {
                                    if (!person.isSeizureConfirmed()) {
                                        continue;
                                    }
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
                                        
                                        snapshotListener.accept(new PersonSnapshot(copy, now, cameraName));
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
                connectionError = formatConnectionError(e.getMessage());
                repaint();
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
            double maxIoU = 0.0;
            
            for (TrackedPerson p : trackedPeople) {
                double minIoU = p.isDetectedInCurrentFrame() ? 0.3 : 0.15;
                double iou = calculateIoU(det.bounds, p.getBounds());
                if (iou > minIoU && iou > maxIoU) {
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

            // Feed normalised skeletal frame into the transformer buffer
            bestMatch.addSkeletalFrame(det.keypoints, fullMat.cols(), fullMat.rows());
        }
        
        // Cleanup old people
        for (TrackedPerson p : trackedPeople) {
            if (p.incrementLastSeen() < 15) {
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
            if (p.isDetectedInCurrentFrame() && p.getLastKeypoints() != null && p.getPrevKeypoints() != null) {
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
                // Expand bounds by 25% if undetected in current frame to capture peripheral thrashing
                int expansionX = p.isDetectedInCurrentFrame() ? 0 : (int) (r.width * 0.125);
                int expansionY = p.isDetectedInCurrentFrame() ? 0 : (int) (r.height * 0.125);
                int x = Math.max(0, r.x - expansionX);
                int y = Math.max(0, r.y - expansionY);
                int w = Math.min(r.width + 2 * expansionX, fullMat.cols() - x);
                int h = Math.min(r.height + 2 * expansionY, fullMat.rows() - y);
                
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

            // ── Transformer Inference ────────────────────────────────────────────
            // Run every frame once the 32-frame skeletal buffer is full.
            // This is cheap (~1-2 ms on GPU) and the result fuses with the
            // FFT-based logic via the updated isSeizureConfirmed() predicate.
            if (p.isSkeletalBufferReady()) {
                float[][] window = p.getSkeletalWindow();
                float prob = seizureDetector.predictProbability(window);
                p.setTransformerSeizureProb(prob);
                p.setTransformerSeizure(prob >= 0.65f);
            }
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
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

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

            g2.drawImage(currentFrame, x, y, newWidth, newHeight, null);

            // Draw Camera Name Overlay
            g2.setColor(new Color(0, 0, 0, 160));
            g2.fillRect(x + 10, y + 10, 180, 25);
            g2.setColor(Color.WHITE);
            g2.setFont(new Font("Arial", Font.BOLD, 12));
            g2.drawString(cameraName, x + 15, y + 27);

            for (TrackedPerson person : renderList) {
                Rectangle rect = person.getBounds();
                int rx = x + (int) (rect.x * ratio);
                int ry = y + (int) (rect.y * ratio);
                int rw = (int) (rect.width * ratio);
                int rh = (int) (rect.height * ratio);
                
                String fftStats = String.format("FFT %.1f Hz (Amp:%.1f Pow:%.1f PAPR:%.1f)",
                        person.getPeakFrequencyHz(), person.getPeakAmplitude(),
                        person.getTotalPower(), person.getPapr());
                String txStats = String.format("TX P(sz)=%.2f", person.getTransformerSeizureProb());

                if (person.isSeizureConfirmed()) {
                    g2.setColor(Color.RED);
                    g2.setStroke(new BasicStroke(3.0f));
                    g2.drawRect(rx, ry, rw, rh);
                    g2.setFont(new Font("Arial", Font.BOLD, 14));
                    g2.drawString("SEIZURE ALARM (CONFIRMED) " + txStats, rx, ry - 20);
                    g2.setFont(new Font("Arial", Font.PLAIN, 11));
                    g2.drawString(fftStats, rx, ry - 7);
                } else if (person.isSeizureWarning()) {
                    g2.setColor(Color.ORANGE);
                    g2.setStroke(new BasicStroke(2.5f));
                    g2.drawRect(rx, ry, rw, rh);
                    g2.setFont(new Font("Arial", Font.BOLD, 12));
                    String warningLabel = person.isTonicWarning() ? "WARNING: TONIC PHASE" : "WARNING: SUSPECTED SEIZURE";
                    g2.drawString(warningLabel + " " + txStats, rx, ry - 20);
                    g2.setFont(new Font("Arial", Font.PLAIN, 11));
                    g2.drawString(fftStats, rx, ry - 7);
                } else {
                    g2.setColor(Color.GREEN);
                    g2.setStroke(new BasicStroke(2.0f));
                    g2.drawRect(rx, ry, rw, rh);
                    g2.setFont(new Font("Arial", Font.PLAIN, 11));
                    g2.drawString(fftStats + " | " + txStats, rx, ry - 5);
                }

                // Render Skeleton Overlay
                if (person.getLastKeypoints() != null) {
                    drawSkeleton(g2, person.getLastKeypoints(), ratio, x, y);
                }
            }
        } else {
            g2.setColor(Color.DARK_GRAY);
            g2.fillRect(0, 0, getWidth(), getHeight());
            g2.setColor(Color.LIGHT_GRAY);
            g2.setFont(new Font("Arial", Font.BOLD, 14));
            String status = cameraName + " (Offline)";
            if (connectionError != null) {
                status = cameraName + " (" + connectionError + ")";
            }
            FontMetrics fm = g2.getFontMetrics();
            int sx = (getWidth() - fm.stringWidth(status)) / 2;
            int sy = (getHeight() - fm.getHeight()) / 2 + fm.getAscent();
            g2.drawString(status, sx, sy);
        }

        // Draw selection border if selected
        if (selected) {
            g2.setColor(new Color(255, 140, 0)); // Dark orange border
            g2.setStroke(new BasicStroke(4.0f));
            g2.drawRect(2, 2, getWidth() - 4, getHeight() - 4);
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

    private String formatConnectionError(String msg) {
        if (msg == null) return "Offline";
        if (msg.contains("-808465656") || msg.contains("-825242872") || msg.contains("401 Unauthorized") || msg.contains("Unauthorized")) {
            return "Offline: Unauthorized (401)";
        }
        if (msg.contains("Connection timed out") || msg.contains("-110") || msg.contains("timeout")) {
            return "Offline: Timeout";
        }
        if (msg.contains("Connection refused") || msg.contains("-111")) {
            return "Offline: Connection Refused";
        }
        if (msg.contains("Host is unreachable") || msg.contains("-113")) {
            return "Offline: Host Unreachable";
        }
        if (msg.contains("404 Not Found") || msg.contains("Server returned 404")) {
            return "Offline: Stream Route Not Found (404)";
        }
        if (msg.contains("400 Bad Request") || msg.contains("Server returned 400")) {
            return "Offline: Bad Request (400)";
        }
        if (msg.contains("avformat_open_input() error")) {
            int idx = msg.indexOf("avformat_open_input()");
            return "Offline: " + msg.substring(idx);
        }
        return "Offline: " + msg;
    }

    public void release() {
        stop();
        if (yoloDetector != null && ownsDetector) {
            yoloDetector.close();
        }
        if (seizureDetector != null && ownsSeizureDetector) {
            seizureDetector.close();
        }
    }
}
