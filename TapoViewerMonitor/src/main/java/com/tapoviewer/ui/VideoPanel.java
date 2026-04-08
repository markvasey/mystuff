package com.tapoviewer.ui;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import com.tapoviewer.model.PersonSnapshot;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.javacpp.FloatPointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.time.LocalDateTime;
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
    private final List<Rectangle> personDetections = new CopyOnWriteArrayList<>();
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
                        hog.detectMultiScale(resizedMat, detections);

                        // Rescale results and update thread-safe list
                        personDetections.clear();
                        for (long i = 0; i < detections.size(); i++) {
                            Rect r = detections.get(i);
                            personDetections.add(new Rectangle(
                                    (int) (r.x() / scale),
                                    (int) (r.y() / scale),
                                    (int) (r.width() / scale),
                                    (int) (r.height() / scale)
                            ));
                        }
                        resizedMat.release();
                    }

                    BufferedImage img = converter.getBufferedImage(frame);
                    if (img != null) {
                        currentFrame = img;
                        
                        // Handle Snapshots
                        if (snapshotListener != null && !personDetections.isEmpty()) {
                            LocalDateTime now = LocalDateTime.now();
                            if (now.isAfter(lastSnapshotTime.plusSeconds(1))) {
                                for (Rectangle rect : personDetections) {
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
                                        break; // Only capture one person per frame to avoid spam
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
            g2.setColor(Color.GREEN);
            g2.setStroke(new BasicStroke(3.0f));
            for (Rectangle rect : personDetections) {
                int rx = x + (int) (rect.x * ratio);
                int ry = y + (int) (rect.y * ratio);
                int rw = (int) (rect.width * ratio);
                int rh = (int) (rect.height * ratio);
                g2.drawRect(rx, ry, rw, rh);
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
        personDetections.clear();
        repaint();
    }

    public void release() {
        stop();
    }
}
