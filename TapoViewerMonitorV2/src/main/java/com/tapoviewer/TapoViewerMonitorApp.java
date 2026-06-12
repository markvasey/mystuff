package com.tapoviewer;

import com.tapoviewer.ui.MainFrame;
import javax.swing.*;

public class TapoViewerMonitorApp {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            MainFrame frame = new MainFrame();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}
