package com.tapoviewer.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

public class MainFrame extends JFrame {
    private final VideoPanel videoPanel;
    private final ControlPanel controlPanel;

    public MainFrame() {
        setTitle("TapoViewer");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1024, 768);
        setLayout(new BorderLayout());

        videoPanel = new VideoPanel();
        controlPanel = new ControlPanel(videoPanel);

        add(videoPanel, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.EAST);

        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                videoPanel.stop();
                videoPanel.release();
            }
        });
    }
}
