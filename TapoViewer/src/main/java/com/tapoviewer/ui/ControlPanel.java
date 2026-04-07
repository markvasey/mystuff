package com.tapoviewer.ui;

import com.tapoviewer.camera.CameraClient;
import com.tapoviewer.model.CameraSettings;
import io.github.hyeonmo.models.ptz.PtzType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.io.InputStream;
import java.util.Properties;

public class ControlPanel extends JPanel {
    private static final Logger logger = LoggerFactory.getLogger(ControlPanel.class);
    private final JTextField ipField = new JTextField("192.168.1.106", 10);
    private final JTextField userField = new JTextField("", 10);
    private final JPasswordField passField = new JPasswordField("", 10);
    private final JCheckBox lowResCheck = new JCheckBox("Low Res (stream2)", false);
    private final JButton connectBtn = new JButton("Connect");

    private CameraClient client;
    private final VideoPanel videoPanel;

    public ControlPanel(VideoPanel videoPanel) {
        this.videoPanel = videoPanel;
        setLayout(new BorderLayout());

        loadSecrets();

        JPanel settingsPanel = new JPanel(new GridLayout(4, 2));
        settingsPanel.add(new JLabel("IP:"));
        settingsPanel.add(ipField);
        settingsPanel.add(new JLabel("Username:"));
        settingsPanel.add(userField);
        settingsPanel.add(new JLabel("Password:"));
        settingsPanel.add(passField);
        settingsPanel.add(lowResCheck);
        settingsPanel.add(connectBtn);

        add(settingsPanel, BorderLayout.NORTH);

        JPanel ptzPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.insets = new Insets(5, 5, 5, 5);

        JButton upBtn = new JButton("▲");
        JButton downBtn = new JButton("▼");
        JButton leftBtn = new JButton("◀");
        JButton rightBtn = new JButton("▶");
        JButton zoomInBtn = new JButton("Zoom +");
        JButton zoomOutBtn = new JButton("Zoom -");

        gbc.gridx = 1; gbc.gridy = 0; ptzPanel.add(upBtn, gbc);
        gbc.gridx = 0; gbc.gridy = 1; ptzPanel.add(leftBtn, gbc);
        gbc.gridx = 2; gbc.gridy = 1; ptzPanel.add(rightBtn, gbc);
        gbc.gridx = 1; gbc.gridy = 2; ptzPanel.add(downBtn, gbc);
        gbc.gridx = 0; gbc.gridy = 3; ptzPanel.add(zoomInBtn, gbc);
        gbc.gridx = 2; gbc.gridy = 3; ptzPanel.add(zoomOutBtn, gbc);

        add(ptzPanel, BorderLayout.CENTER);

        connectBtn.addActionListener(e -> connect());

        upBtn.addActionListener(e -> move(PtzType.UP));
        downBtn.addActionListener(e -> move(PtzType.DOWN));
        leftBtn.addActionListener(e -> move(PtzType.LEFT));
        rightBtn.addActionListener(e -> move(PtzType.RIGHT));
        zoomInBtn.addActionListener(e -> move(PtzType.ZOOM_IN));
        zoomOutBtn.addActionListener(e -> move(PtzType.ZOOM_OUT));
    }

    private void loadSecrets() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream("secret.txt")) {
            if (input == null) {
                logger.warn("secret.txt not found in resources");
                return;
            }
            Properties prop = new Properties();
            prop.load(input);
            userField.setText(prop.getProperty("username", ""));
            passField.setText(prop.getProperty("password", ""));
            logger.info("Loaded credentials from secret.txt");
        } catch (Exception ex) {
            logger.error("Error loading secret.txt", ex);
        }
    }

    private void connect() {
        CameraSettings settings = new CameraSettings();
        // Trim inputs to remove any invisible trailing spaces!
        settings.setIp(ipField.getText().trim());
        settings.setOnvifPort(2020);
        settings.setRtspUsername(userField.getText().trim());
        settings.setRtspPassword(new String(passField.getPassword()).trim());
        settings.setOnvifUsername(userField.getText().trim());
        settings.setOnvifPassword(new String(passField.getPassword()).trim());

        logger.info("Connecting to IP: {} via ONVIF (port 2020)...", settings.getIp());
        client = new CameraClient(settings);
        client.connect().thenRun(() -> {
            logger.info("ONVIF Connection Successful. Handing off to RTSP engine...");
            SwingUtilities.invokeLater(() -> {
                String stream = lowResCheck.isSelected() ? "stream2" : "stream1";
                videoPanel.play(
                        settings.getIp(),
                        settings.getRtspPort(),
                        stream,
                        settings.getRtspUsername(),
                        settings.getRtspPassword()
                );
                connectBtn.setText("Connected");
                connectBtn.setEnabled(false);
            });
        }).exceptionally(ex -> {
            logger.error("ONVIF Connection Failed: {}", ex.getMessage());
            SwingUtilities.invokeLater(() -> {
                JOptionPane.showMessageDialog(this, "Failed to connect: " + ex.getMessage());
            });
            return null;
        });
    }

    private void move(PtzType type) {
        if (client != null) {
            client.ptzMove(type).exceptionally(ex -> {
                logger.error("PTZ move failed", ex);
                return null;
            });
        }
    }
}
