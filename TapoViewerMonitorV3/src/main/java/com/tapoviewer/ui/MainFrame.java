package com.tapoviewer.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class MainFrame extends JFrame {
    private final VideoGridPanel videoGridPanel;
    private final ControlPanel controlPanel;

    public MainFrame() {
        setTitle("TapoViewer Multi-Camera Monitor");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1280, 800); // Increased width to comfortably fit multiple tiled streams
        setLayout(new BorderLayout());

        videoGridPanel = new VideoGridPanel();
        controlPanel = new ControlPanel(videoGridPanel);

        add(videoGridPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.EAST);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                videoGridPanel.stop();
                videoGridPanel.release();
                controlPanel.release();
            }
        });
    }
}
