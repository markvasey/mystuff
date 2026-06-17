package com.tapoviewer.ui;

import com.tapoviewer.math.SeizureDetector;
import com.tapoviewer.math.YoloPoseDetector;
import com.tapoviewer.model.PersonSnapshot;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

public class VideoGridPanel extends JPanel {
    private static final Logger logger = LoggerFactory.getLogger(VideoGridPanel.class);

    private final List<VideoPanel> videoPanels = new ArrayList<>();
    private final YoloPoseDetector sharedDetector;
    private final SeizureDetector sharedSeizureDetector;
    private Consumer<PersonSnapshot> snapshotListener;
    private Consumer<String> selectionListener;
    private String selectedCameraName = null;

    public VideoGridPanel() {
        setLayout(new GridLayout(1, 1, 5, 5)); // Default to single cell
        setBackground(Color.BLACK);
        // One shared YoloPoseDetector and SeizureDetector across all camera feeds.
        // Both are thread-safe (synchronized on their OrtSession) so all VideoPanel
        // worker threads can call them concurrently without contention.
        sharedDetector = new YoloPoseDetector();
        sharedSeizureDetector = new SeizureDetector();
    }

    public void setSnapshotListener(Consumer<PersonSnapshot> listener) {
        this.snapshotListener = listener;
        for (VideoPanel panel : videoPanels) {
            panel.setSnapshotListener(listener);
        }
    }

    public void setSelectionListener(Consumer<String> listener) {
        this.selectionListener = listener;
    }

    public void playCameras(Map<String, String> cameras, String stream, String username, String password, boolean useGpu) {
        stop(); // Stop any running feeds
        removeAll();
        videoPanels.clear();

        int numCameras = cameras.size();
        if (numCameras == 0) {
            setLayout(new GridLayout(1, 1));
            revalidate();
            repaint();
            return;
        }

        // Dynamically compute rows and columns for a nice grid layout
        int cols = (int) Math.ceil(Math.sqrt(numCameras));
        int rows = (int) Math.ceil((double) numCameras / cols);
        setLayout(new GridLayout(rows, cols, 5, 5));

        for (Map.Entry<String, String> entry : cameras.entrySet()) {
            String name = entry.getKey();
            String ip = entry.getValue();

            VideoPanel panel = new VideoPanel(sharedDetector, sharedSeizureDetector);
            panel.setSnapshotListener(snapshotListener);
            
            // Add click selection listener to panel
            panel.addMouseListener(new java.awt.event.MouseAdapter() {
                @Override
                public void mousePressed(java.awt.event.MouseEvent e) {
                    setSelectedCamera(name);
                    if (selectionListener != null) {
                        selectionListener.accept(name);
                    }
                }
            });

            add(panel);
            videoPanels.add(panel);

            // Start playing the video stream
            panel.play(name, ip, 554, stream, username, password, useGpu);
        }

        revalidate();
        repaint();
    }

    public void setSelectedCamera(String cameraName) {
        this.selectedCameraName = cameraName;
        for (VideoPanel panel : videoPanels) {
            panel.setSelected(panel.getCameraName().equals(cameraName));
        }
    }

    public void stop() {
        for (VideoPanel panel : videoPanels) {
            panel.stop();
        }
    }

    public void release() {
        stop();
        for (VideoPanel panel : videoPanels) {
            panel.release();
        }
        videoPanels.clear();
        if (sharedDetector != null) {
            sharedDetector.close();
        }
        if (sharedSeizureDetector != null) {
            sharedSeizureDetector.close();
        }
    }
}
