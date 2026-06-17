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
import java.util.concurrent.CompletableFuture;

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
    private final VideoGridPanel videoGridPanel;
    private final DetectionHistoryPanel historyPanel = new DetectionHistoryPanel();
    private boolean isAutoConnecting = false;
    
    private final Map<String, Map<String, String>> houseData = new LinkedHashMap<>();
    private String defaultHouse = null;
    private String defaultCamera = null;

    public ControlPanel(VideoGridPanel videoGridPanel) {
        this.videoGridPanel = videoGridPanel;
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
        videoGridPanel.setSnapshotListener(historyPanel::addSnapshot);

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

        // Register selection listener
        videoGridPanel.setSelectionListener(cameraName -> {
            isAutoConnecting = true;
            cameraCombo.setSelectedItem(cameraName);
            isAutoConnecting = false;
            
            String selectedHouse = (String) houseCombo.getSelectedItem();
            if (selectedHouse != null) {
                String ip = houseData.get(selectedHouse).get(cameraName);
                if (ip != null) {
                    ipLabel.setText(ip);
                    connectPtz(cameraName, ip);
                }
            }
        });

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
            isAutoConnecting = true;
            String selectedHouse = (String) houseCombo.getSelectedItem();
            cameraCombo.removeAllItems();
            if (selectedHouse != null && houseData.containsKey(selectedHouse)) {
                Map<String, String> cameras = houseData.get(selectedHouse);
                for (String cam : cameras.keySet()) {
                    cameraCombo.addItem(cam);
                }
            }
            isAutoConnecting = false;
        });

        cameraCombo.addActionListener(e -> {
            if (isAutoConnecting) return;
            String selectedHouse = (String) houseCombo.getSelectedItem();
            String selectedCam = (String) cameraCombo.getSelectedItem();
            if (selectedHouse != null && selectedCam != null) {
                String ip = houseData.get(selectedHouse).get(selectedCam);
                ipLabel.setText(ip);
                
                if (connectBtn.getText().equals("Disconnect") || connectBtn.getText().equals("Connected")) {
                    videoGridPanel.setSelectedCamera(selectedCam);
                    connectPtz(selectedCam, ip);
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

    private boolean isPortReachable(String ip, int port, int timeoutMs) {
        try (java.net.Socket socket = new java.net.Socket()) {
            socket.connect(new java.net.InetSocketAddress(ip, port), timeoutMs);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private void connectPtz(String cameraName, String ip) {
        if (client != null) {
            client.release();
            client = null;
        }
        setPtzEnabled(false, false);

        CameraSettings settings = new CameraSettings();
        settings.setIp(ip.trim());
        settings.setRtspUsername(userField.getText().trim());
        settings.setRtspPassword(new String(passField.getPassword()).trim());
        settings.setOnvifUsername(userField.getText().trim());
        settings.setOnvifPassword(new String(passField.getPassword()).trim());

        client = new CameraClient(settings);
        client.connect().thenAccept(zoomSupported -> {
            SwingUtilities.invokeLater(() -> {
                setPtzEnabled(true, zoomSupported);
                logger.info("PTZ connected for camera: {}", cameraName);
            });
        }).exceptionally(ex -> {
            logger.warn("ONVIF PTZ Connection Failed for camera {}: {}", cameraName, ex.getMessage());
            return null;
        });
    }

    private void connect() {
        if (client != null) {
            client.release();
            client = null;
        }
        videoGridPanel.stop();
        setPtzEnabled(false, false);

        String selectedHouse = (String) houseCombo.getSelectedItem();
        if (selectedHouse == null) return;

        Map<String, String> houseCameras = houseData.get(selectedHouse);
        if (houseCameras == null || houseCameras.isEmpty()) {
            JOptionPane.showMessageDialog(this, "No cameras configured for this house.");
            return;
        }

        connectBtn.setEnabled(false);
        connectBtn.setText("Connecting...");

        // Perform discovery (reachability check) in parallel
        CompletableFuture.supplyAsync(() -> {
            Map<String, String> connectable = new LinkedHashMap<>();
            java.util.List<CompletableFuture<Map.Entry<String, String>>> futures = new ArrayList<>();
            for (Map.Entry<String, String> entry : houseCameras.entrySet()) {
                futures.add(CompletableFuture.supplyAsync(() -> {
                    boolean ok = isPortReachable(entry.getValue(), 554, 1500);
                    return ok ? entry : null;
                }));
            }
            for (CompletableFuture<Map.Entry<String, String>> f : futures) {
                try {
                    Map.Entry<String, String> entry = f.get();
                    if (entry != null) {
                        connectable.put(entry.getKey(), entry.getValue());
                    }
                } catch (Exception e) {
                    // Ignore
                }
            }
            return connectable;
        }).thenAccept(connectable -> {
            SwingUtilities.invokeLater(() -> {
                if (connectable.isEmpty()) {
                    connectBtn.setText("Connect");
                    connectBtn.setEnabled(true);
                    JOptionPane.showMessageDialog(this, "No connectable cameras discovered.");
                    return;
                }

                String stream = lowResCheck.isSelected() ? "stream2" : "stream1";
                boolean useGpu = gpuDecodeCheck.isSelected();
                String username = userField.getText().trim();
                String password = new String(passField.getPassword()).trim();

                videoGridPanel.playCameras(connectable, stream, username, password, useGpu);

                connectBtn.setText("Connected");
                connectBtn.setEnabled(true);

                // Default to selecting the first connectable camera
                String firstCam = connectable.keySet().iterator().next();
                cameraCombo.setSelectedItem(firstCam);

                // Transition button to "Disconnect" state after 1 second delay
                javax.swing.Timer timer = new javax.swing.Timer(1000, evt -> connectBtn.setText("Disconnect"));
                timer.setRepeats(false);
                timer.start();
            });
        }).exceptionally(ex -> {
            logger.error("ONVIF/Video Connection Failed: {}", ex.getMessage());
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
        videoGridPanel.stop();
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
