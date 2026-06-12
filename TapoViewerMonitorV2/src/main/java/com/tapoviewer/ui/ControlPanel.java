package com.tapoviewer.ui;

import com.tapoviewer.camera.CameraClient;
import com.tapoviewer.model.CameraSettings;
import io.github.hyeonmo.models.ptz.PtzType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import javax.swing.*;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import java.awt.*;
import java.io.InputStream;
import java.util.*;

public class ControlPanel extends JPanel {
    private static final Logger logger = LoggerFactory.getLogger(ControlPanel.class);
    
    private final JComboBox<String> houseCombo = new JComboBox<>();
    private final JComboBox<String> cameraCombo = new JComboBox<>();
    private final JLabel ipLabel = new JLabel("0.0.0.0");
    
    private final JTextField userField = new JTextField("", 10);
    private final JPasswordField passField = new JPasswordField("", 10);
    private final JCheckBox lowResCheck = new JCheckBox("Low Res (stream2)", false);
    private final JCheckBox gpuDecodeCheck = new JCheckBox("GPU Decode (NVDEC)", false);
    private final JButton connectBtn = new JButton("Connect");

    private final JButton upBtn = new JButton("▲");
    private final JButton downBtn = new JButton("▼");
    private final JButton leftBtn = new JButton("◀");
    private final JButton rightBtn = new JButton("▶");
    private final JButton zoomInBtn = new JButton("Zoom +");
    private final JButton zoomOutBtn = new JButton("Zoom -");

    private CameraClient client;
    private final VideoPanel videoPanel;
    private final DetectionHistoryPanel historyPanel = new DetectionHistoryPanel();
    private boolean isAutoConnecting = false;
    
    private final Map<String, Map<String, String>> houseData = new LinkedHashMap<>();
    private String defaultHouse = null;
    private String defaultCamera = null;

    public ControlPanel(VideoPanel videoPanel) {
        this.videoPanel = videoPanel;
        setLayout(new BorderLayout());

        loadCameraXml();
        loadSecrets();
        setupCombos();

        // Initialize GPU Decode Checkbox based on NVIDIA GPU presence
        boolean gpuPresent = isNvidiaGpuPresent();
        gpuDecodeCheck.setSelected(gpuPresent);
        gpuDecodeCheck.setEnabled(gpuPresent);
        if (!gpuPresent) {
            gpuDecodeCheck.setToolTipText("NVIDIA eGPU not detected. NVDEC is disabled.");
        } else {
            gpuDecodeCheck.setToolTipText("Hardware-accelerated video decoding using NVIDIA NVDEC.");
        }

        // Register snapshot listener
        videoPanel.setSnapshotListener(historyPanel::addSnapshot);

        JPanel settingsPanel = new JPanel(new GridLayout(7, 2));
        settingsPanel.add(new JLabel("House:"));
        settingsPanel.add(houseCombo);
        settingsPanel.add(new JLabel("Camera:"));
        settingsPanel.add(cameraCombo);
        settingsPanel.add(new JLabel("IP:"));
        settingsPanel.add(ipLabel);
        settingsPanel.add(new JLabel("Username:"));
        settingsPanel.add(userField);
        settingsPanel.add(new JLabel("Password:"));
        settingsPanel.add(passField);
        settingsPanel.add(lowResCheck);
        settingsPanel.add(gpuDecodeCheck);
        settingsPanel.add(new JLabel("")); // spacer
        settingsPanel.add(connectBtn);

        add(settingsPanel, BorderLayout.NORTH);

        JPanel ptzPanel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.BOTH;
        gbc.insets = new Insets(5, 5, 5, 5);

        gbc.gridx = 1; gbc.gridy = 0; ptzPanel.add(upBtn, gbc);
        gbc.gridx = 0; gbc.gridy = 1; ptzPanel.add(leftBtn, gbc);
        gbc.gridx = 2; gbc.gridy = 1; ptzPanel.add(rightBtn, gbc);
        gbc.gridx = 1; gbc.gridy = 2; ptzPanel.add(downBtn, gbc);
        gbc.gridx = 0; gbc.gridy = 3; ptzPanel.add(zoomInBtn, gbc);
        gbc.gridx = 2; gbc.gridy = 3; ptzPanel.add(zoomOutBtn, gbc);

        add(ptzPanel, BorderLayout.CENTER);
        add(historyPanel, BorderLayout.SOUTH);

        connectBtn.addActionListener(e -> {
            if (connectBtn.getText().equals("Disconnect")) {
                disconnect();
            } else {
                connect();
            }
        });

        upBtn.addActionListener(e -> move(PtzType.UP));
        downBtn.addActionListener(e -> move(PtzType.DOWN));
        leftBtn.addActionListener(e -> move(PtzType.LEFT));
        rightBtn.addActionListener(e -> move(PtzType.RIGHT));
        zoomInBtn.addActionListener(e -> move(PtzType.ZOOM_IN));
        zoomOutBtn.addActionListener(e -> move(PtzType.ZOOM_OUT));
        
        // Initially disable PTZ until connected
        setPtzEnabled(false, false);

        // Automatically connect on startup
        SwingUtilities.invokeLater(this::connect);
    }

    private void setPtzEnabled(boolean ptzEnabled, boolean zoomEnabled) {
        upBtn.setEnabled(ptzEnabled);
        downBtn.setEnabled(ptzEnabled);
        leftBtn.setEnabled(ptzEnabled);
        rightBtn.setEnabled(ptzEnabled);
        zoomInBtn.setEnabled(ptzEnabled && zoomEnabled);
        zoomOutBtn.setEnabled(ptzEnabled && zoomEnabled);
    }

    private void loadCameraXml() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream("cameras.xml")) {
            if (input == null) {
                logger.warn("cameras.xml not found in resources");
                return;
            }
            DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            Document doc = dBuilder.parse(input);
            doc.getDocumentElement().normalize();

            NodeList houseList = doc.getElementsByTagName("House");
            for (int i = 0; i < houseList.getLength(); i++) {
                Element houseEl = (Element) houseList.item(i);
                String houseName = houseEl.getAttribute("name");
                if ("true".equals(houseEl.getAttribute("default"))) {
                    defaultHouse = houseName;
                }
                Map<String, String> cameras = new LinkedHashMap<>();
                
                NodeList camList = houseEl.getElementsByTagName("Camera");
                for (int j = 0; j < camList.getLength(); j++) {
                    Element camEl = (Element) camList.item(j);
                    String camName = camEl.getAttribute("name");
                    cameras.put(camName, camEl.getAttribute("ip"));
                    if ("true".equals(camEl.getAttribute("default"))) {
                        defaultCamera = camName;
                    }
                }
                houseData.put(houseName, cameras);
            }
            logger.info("Loaded {} houses from cameras.xml", houseData.size());
        } catch (Exception ex) {
            logger.error("Error loading cameras.xml", ex);
        }
    }

    private void setupCombos() {
        isAutoConnecting = true;
        for (String house : houseData.keySet()) {
            houseCombo.addItem(house);
        }

        houseCombo.addActionListener(e -> {
            String selectedHouse = (String) houseCombo.getSelectedItem();
            cameraCombo.removeAllItems();
            if (selectedHouse != null && houseData.containsKey(selectedHouse)) {
                Map<String, String> cameras = houseData.get(selectedHouse);
                for (String cam : cameras.keySet()) {
                    cameraCombo.addItem(cam);
                }
            }
        });

        cameraCombo.addActionListener(e -> {
            String selectedHouse = (String) houseCombo.getSelectedItem();
            String selectedCam = (String) cameraCombo.getSelectedItem();
            if (selectedHouse != null && selectedCam != null) {
                String ip = houseData.get(selectedHouse).get(selectedCam);
                ipLabel.setText(ip);
                
                if (!isAutoConnecting && (connectBtn.getText().equals("Disconnect") || connectBtn.getText().equals("Connected"))) {
                    connect();
                }
            }
        });

        if (defaultHouse != null && houseData.containsKey(defaultHouse)) {
            houseCombo.setSelectedItem(defaultHouse);
            if (defaultCamera != null && houseData.get(defaultHouse).containsKey(defaultCamera)) {
                cameraCombo.setSelectedItem(defaultCamera);
            }
        } else if (houseCombo.getItemCount() > 0) {
            houseCombo.setSelectedIndex(0);
        }
        isAutoConnecting = false;
    }

    private void loadSecrets() {
        try (InputStream input = getClass().getClassLoader().getResourceAsStream("secret.txt")) {
            if (input == null) return;
            Properties prop = new Properties();
            prop.load(input);
            userField.setText(prop.getProperty("username", ""));
            passField.setText(prop.getProperty("password", ""));
        } catch (Exception ex) {
            logger.error("Error loading secret.txt", ex);
        }
    }

    private void connect() {
        if (client != null) {
            videoPanel.stop();
            client.release();
        }

        setPtzEnabled(false, false);

        CameraSettings settings = new CameraSettings();
        settings.setIp(ipLabel.getText().trim());
        settings.setRtspUsername(userField.getText().trim());
        settings.setRtspPassword(new String(passField.getPassword()).trim());
        settings.setOnvifUsername(userField.getText().trim());
        settings.setOnvifPassword(new String(passField.getPassword()).trim());

        connectBtn.setEnabled(false);
        connectBtn.setText("Connecting...");

        client = new CameraClient(settings);
        client.connect().thenAccept(zoomSupported -> {
            SwingUtilities.invokeLater(() -> {
                String stream = lowResCheck.isSelected() ? "stream2" : "stream1";
                boolean useGpu = gpuDecodeCheck.isSelected();
                videoPanel.play(
                        settings.getIp(),
                        settings.getRtspPort(),
                        stream,
                        settings.getRtspUsername(),
                        settings.getRtspPassword(),
                        useGpu
                );
                connectBtn.setText("Connected");
                connectBtn.setEnabled(true);
                setPtzEnabled(true, zoomSupported);

                // Transition to Disconnect after a 1-second delay
                javax.swing.Timer timer = new javax.swing.Timer(1000, evt -> connectBtn.setText("Disconnect"));
                timer.setRepeats(false);
                timer.start();
            });
        }).exceptionally(ex -> {
            logger.error("ONVIF Connection Failed: {}", ex.getMessage());
            SwingUtilities.invokeLater(() -> {
                connectBtn.setEnabled(true);
                connectBtn.setText("Connect");
                JOptionPane.showMessageDialog(this, "Failed to connect: " + ex.getMessage());
            });
            return null;
        });
    }

    private void disconnect() {
        if (client != null) {
            client.release();
            client = null;
        }
        videoPanel.stop();
        connectBtn.setText("Connect");
        setPtzEnabled(false, false);
    }

    public void release() {
        if (client != null) {
            client.release();
        }
    }

    private void move(PtzType type) {
        if (client != null) {
            client.ptzMove(type).exceptionally(ex -> {
                logger.error("PTZ move failed", ex);
                return null;
            });
        }
    }

    private boolean isNvidiaGpuPresent() {
        if (new java.io.File("/dev/nvidia0").exists()) {
            return true;
        }
        try {
            Process process = new ProcessBuilder("nvidia-smi").start();
            return process.waitFor() == 0;
        } catch (Exception e) {
            return false;
        }
    }
}
