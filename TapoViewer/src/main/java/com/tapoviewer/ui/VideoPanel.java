package com.tapoviewer.ui;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class VideoPanel extends JPanel {
    private static final Logger logger = LoggerFactory.getLogger(VideoPanel.class);
    private FFmpegFrameGrabber grabber;
    private volatile boolean running = false;
    private BufferedImage currentFrame;
    private Thread workerThread;

    public VideoPanel() {
        setBackground(Color.BLACK);
        setLayout(new BorderLayout());
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
                    
                    BufferedImage img = converter.getBufferedImage(frame);
                    if (img != null) {
                        currentFrame = img;
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
        repaint();
    }

    public void release() {
        stop();
    }
}
