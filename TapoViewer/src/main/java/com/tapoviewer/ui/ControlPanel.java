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
import java.util.List;

public class ControlPanel extends JPanel {
    private static final Logger logger = LoggerFactory.getLogger(ControlPanel.class);
    
    private final JComboBox<String> houseCombo = new JComboBox<>();
    private final JComboBox<String> cameraCombo = new JComboBox<>();
    private final JLabel ipLabel = new JLabel("0.0.0.0");
    
    private final JTextField userField = new JTextField("", 10);
    private final JPasswordField passField = new JPasswordField("", 10);
    private final JCheckBox lowResCheck = new JCheckBox("Low Res (stream2)", false);
    private final JButton connectBtn = new JButton("Connect");

    private CameraClient client;
    private final VideoPanel videoPanel;
    
    private final Map<String, Map<String, String>> houseData = new LinkedHashMap<>();

    public ControlPanel(VideoPanel videoPanel) {
        this.videoPanel = videoPanel;
        setLayout(new BorderLayout());

        loadCameraXml();
        loadSecrets();
        setupCombos();

        JPanel settingsPanel = new JPanel(new GridLayout(6, 2));
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
                Map<String, String> cameras = new LinkedHashMap<>();
                
                NodeList camList = houseEl.getElementsByTagName("Camera");
                for (int j = 0; j < camList.getLength(); j++) {
                    Element camEl = (Element) camList.item(j);
                    cameras.put(camEl.getAttribute("name"), camEl.getAttribute("ip"));
                }
                houseData.put(houseName, cameras);
            }
            logger.info("Loaded {} houses from cameras.xml", houseData.size());
        } catch (Exception ex) {
            logger.error("Error loading cameras.xml", ex);
        }
    }

    private void setupCombos() {
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
            }
        });

        // Trigger initial load
        if (houseCombo.getItemCount() > 0) {
            houseCombo.setSelectedIndex(0);
        }
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
        settings.setIp(ipLabel.getText().trim());
        settings.setOnvifPort(2020);
        settings.setRtspUsername(userField.getText().trim());
        settings.setRtspPassword(new String(passField.getPassword()).trim());
        settings.setOnvifUsername(userField.getText().trim());
        settings.setOnvifPassword(new String(passField.getPassword()).trim());

        logger.info("Connecting to IP: {} via ONVIF (port 2020)...", settings.getIp());
        
        // Disable connect button during attempt
        connectBtn.setEnabled(false);
        connectBtn.setText("Connecting...");

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

    private void move(PtzType type) {
        if (client != null) {
            client.ptzMove(type).exceptionally(ex -> {
                logger.error("PTZ move failed", ex);
                return null;
            });
        }
    }
}
