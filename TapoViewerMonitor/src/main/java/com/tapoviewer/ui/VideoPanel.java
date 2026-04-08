package com.tapoviewer.ui;

import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import com.tapoviewer.model.PersonSnapshot;
import com.tapoviewer.model.TrackedPerson;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_video.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.javacpp.FloatPointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    private final HOGDescriptor hog;
    private final OpenCVFrameConverter.ToMat toMatConverter = new OpenCVFrameConverter.ToMat();
    private final List<TrackedPerson> trackedPeople = new ArrayList<>();
    private final List<TrackedPerson> renderList = new CopyOnWriteArrayList<>();
    private Consumer<PersonSnapshot> snapshotListener;
    private LocalDateTime lastSnapshotTime = LocalDateTime.MIN;

    public VideoPanel() {
        setBackground(Color.BLACK);
        setLayout(new BorderLayout());

        // Initialize HOG detector
        hog = new HOGDescriptor();
        hog.setSVMDetector(new Mat(HOGDescriptor.getDefaultPeopleDetector()));
    }

    public void setSnapshotListener(Consumer<PersonSnapshot> listener) {
        this.snapshotListener = listener;
    }

    public void play(String ip, int port, String stream, String username, String password) {
        stop(); // Ensure any previous stream is stopped

        // Suppress FFmpeg noise (including the swscaler deprecated pixel format warning)
        org.bytedeco.ffmpeg.global.avutil.av_log_set_level(org.bytedeco.ffmpeg.global.avutil.AV_LOG_ERROR);

        String mrl = String.format("rtsp://%s:%s@%s:%d/%s", username, password, ip, port, stream);
        logger.info("Connecting to RTSP via JavaCV: rtsp://{}:****@{}:{}/{}", username, ip, port, stream);

        running = true;
        workerThread = new Thread(() -> {
            try {
                grabber = new FFmpegFrameGrabber(mrl);
                grabber.setOption("rtsp_transport", "tcp"); // Force TCP for stability
                grabber.setOption("stimeout", "3000000");   // 3 second timeout
                grabber.setVideoBitrate(0);                 // Auto
                grabber.start();

                Java2DFrameConverter converter = new Java2DFrameConverter();
                while (running) {
                    Frame frame = grabber.grabImage();
                    if (frame == null) break;

                    // Person Detection
                    Mat mat = toMatConverter.convert(frame);
                    if (mat != null && !mat.empty()) {
                        // Downscale for performance (640px wide)
                        double scale = 640.0 / mat.cols();
                        Mat resizedMat = new Mat();
                        org.bytedeco.opencv.global.opencv_imgproc.resize(mat, resizedMat, new Size(640, (int) (mat.rows() * scale)));

                        RectVector detections = new RectVector();
                        // BUG FIX #2: Increased hitThreshold to 0.2 to reduce initial false positives
                        hog.detectMultiScale(resizedMat, detections, 0.3, new Size(8, 8), new Size(32, 32), 1.05, 2.0, false);

                        // Rescale results and update tracking
                        List<Rectangle> currentDetections = new ArrayList<>();
                        for (long i = 0; i < detections.size(); i++) {
                            Rect r = detections.get(i);
                            currentDetections.add(new Rectangle(
                                    (int) (r.x() / scale),
                                    (int) (r.y() / scale),
                                    (int) (r.width() / scale),
                                    (int) (r.height() / scale)
                            ));
                        }
                        
                        updateTracking(mat, currentDetections);
                        resizedMat.release();
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
                                    // Ensure within image bounds
                                    int x = Math.max(0, rect.x);
                                    int y = Math.max(0, rect.y);
                                    int w = Math.min(rect.width, img.getWidth() - x);
                                    int h = Math.min(rect.height, img.getHeight() - y);
                                    
                                    if (w > 0 && h > 0) {
                                        BufferedImage crop = img.getSubimage(x, y, w, h);
                                        // Send a copy to avoid modification issues
                                        BufferedImage copy = new BufferedImage(w, h, crop.getType());
                                        Graphics2D g2 = copy.createGraphics();
                                        g2.drawImage(crop, 0, 0, null);
                                        g2.dispose();
                                        
                                        snapshotListener.accept(new PersonSnapshot(copy, now));
                                        lastSnapshotTime = now;
                                        break; // Only capture one person per frame
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

    private void updateTracking(Mat fullMat, List<Rectangle> detections) {
        // Simple IoU-based tracking
        List<TrackedPerson> matchedPeople = new ArrayList<>();
        
        for (Rectangle det : detections) {
            TrackedPerson bestMatch = null;
            double maxIoU = 0.3; // Minimum overlap threshold
            
            for (TrackedPerson p : trackedPeople) {
                double iou = calculateIoU(det, p.getBounds());
                if (iou > maxIoU) {
                    maxIoU = iou;
                    bestMatch = p;
                }
            }
            
            if (bestMatch != null) {
                bestMatch.setBounds(det);
                bestMatch.resetLastSeen();
                trackedPeople.remove(bestMatch);
                matchedPeople.add(bestMatch);
            } else {
                matchedPeople.add(new TrackedPerson(det));
            }
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
        
        // Calculate Optical Flow for each matched person
        for (TrackedPerson p : trackedPeople) {
            Rectangle r = p.getBounds();
            // Bound safety
            int x = Math.max(0, r.x);
            int y = Math.max(0, r.y);
            int w = Math.min(r.width, fullMat.cols() - x);
            int h = Math.min(r.height, fullMat.rows() - y);
            
            if (w <= 0 || h <= 0) continue;

            Mat region = new Mat(fullMat, new Rect(x, y, w, h));
            Mat grayRegion = new Mat();
            cvtColor(region, grayRegion, COLOR_BGR2GRAY);
            
            if (p.getLastGrayRegion() != null && 
                p.getLastGrayRegion().rows() == grayRegion.rows() && 
                p.getLastGrayRegion().cols() == grayRegion.cols()) {
                
                Mat flow = new Mat();
                calcOpticalFlowFarneback(p.getLastGrayRegion(), grayRegion, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
                
                // Average motion magnitude
                MatVector flowChannels = new MatVector();
                split(flow, flowChannels);
                
                if (flowChannels.size() >= 2) {
                    Mat flowX = flowChannels.get(0);
                    Mat flowY = flowChannels.get(1);
                    Mat mag = new Mat();
                    cartToPolar(flowX, flowY, mag, new Mat());
                    
                    Scalar meanMag = mean(mag);
                    if (meanMag != null) {
                        p.addMotion(meanMag.get());
                    }
                    
                    mag.release();
                    flowX.release();
                    flowY.release();
                }
                
                flowChannels.close();
                flow.release();
            }
            
            p.setLastGrayRegion(grayRegion);
            region.release();
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
            // Scale image to fit panel while preserving aspect ratio
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

            // Draw detections
            Graphics2D g2 = (Graphics2D) g;
            g2.setStroke(new BasicStroke(3.0f));
            for (TrackedPerson person : renderList) {
                Rectangle rect = person.getBounds();
                int rx = x + (int) (rect.x * ratio);
                int ry = y + (int) (rect.y * ratio);
                int rw = (int) (rect.width * ratio);
                int rh = (int) (rect.height * ratio);
                
                if (person.isSeizureDetected()) {
                    g2.setColor(Color.RED);
                    g2.drawRect(rx, ry, rw, rh);
                    g2.setFont(new Font("Arial", Font.BOLD, 16));
                    g2.drawString("SEIZURE DETECTED", rx, ry - 5);
                } else {
                    g2.setColor(Color.GREEN);
                    g2.drawRect(rx, ry, rw, rh);
                }
            }
        } else {
            g.setColor(Color.WHITE);
            g.drawString("No Video Signal", getWidth() / 2 - 40, getHeight() / 2);
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
    }
}
