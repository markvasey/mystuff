package com.tapoviewer.ui;

import com.tapoviewer.model.PersonSnapshot;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

public class DetectionHistoryPanel extends JPanel {
    private final DefaultComboBoxModel<PersonSnapshot> model = new DefaultComboBoxModel<>();
    private final JComboBox<PersonSnapshot> snapshotCombo = new JComboBox<>(model);
    private final JPanel imageDisplay = new JPanel() {
        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            PersonSnapshot selected = (PersonSnapshot) snapshotCombo.getSelectedItem();
            if (selected != null) {
                // Scale image to fit panel
                int pw = getWidth();
                int ph = getHeight();
                int iw = selected.image().getWidth();
                int ih = selected.image().getHeight();
                double ratio = Math.min((double) pw / iw, (double) ph / ih);
                int nw = (int) (iw * ratio);
                int nh = (int) (ih * ratio);
                int x = (pw - nw) / 2;
                int y = (ph - nh) / 2;
                g.drawImage(selected.image(), x, y, nw, nh, null);
            }
        }
    };

    public DetectionHistoryPanel() {
        setLayout(new BorderLayout());
        setBorder(BorderFactory.createTitledBorder("Recent Detections (Last 20)"));

        snapshotCombo.addActionListener(e -> imageDisplay.repaint());
        
        // Ensure combo box handles arrow keys and repaints
        snapshotCombo.setFocusable(true);

        // Clicking the image display should focus the combo for arrow key navigation
        imageDisplay.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                snapshotCombo.requestFocusInWindow();
            }
        });

        add(snapshotCombo, BorderLayout.NORTH);
        imageDisplay.setPreferredSize(new Dimension(200, 200));
        imageDisplay.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        add(imageDisplay, BorderLayout.CENTER);
    }

    public void addSnapshot(PersonSnapshot snapshot) {
        SwingUtilities.invokeLater(() -> {
            model.insertElementAt(snapshot, 0);
            if (model.getSize() > 20) {
                model.removeElementAt(20);
            }
            if (model.getSize() == 1) {
                snapshotCombo.setSelectedIndex(0);
            }
            imageDisplay.repaint();
        });
    }
}
