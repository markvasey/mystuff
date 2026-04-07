package com.tapoviewer.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

@JsonIgnoreProperties(ignoreUnknown = true)
public class CameraSettings {
    private String name = "Tapo Camera";
    private String ip;
    private int rtspPort = 554;
    private int onvifPort = 2020;
    private String rtspUsername;
    private String rtspPassword;
    private String onvifUsername;
    private String onvifPassword;

    public CameraSettings() {}

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getIp() { return ip; }
    public void setIp(String ip) { this.ip = ip; }

    public int getRtspPort() { return rtspPort; }
    public void setRtspPort(int rtspPort) { this.rtspPort = rtspPort; }

    public int getOnvifPort() { return onvifPort; }
    public void setOnvifPort(int onvifPort) { this.onvifPort = onvifPort; }

    public String getRtspUsername() { return rtspUsername; }
    public void setRtspUsername(String rtspUsername) { this.rtspUsername = rtspUsername; }

    public String getRtspPassword() { return rtspPassword; }
    public void setRtspPassword(String rtspPassword) { this.rtspPassword = rtspPassword; }

    public String getOnvifUsername() { return onvifUsername; }
    public void setOnvifUsername(String onvifUsername) { this.onvifUsername = onvifUsername; }

    public String getOnvifPassword() { return onvifPassword; }
    public void setOnvifPassword(String onvifPassword) { this.onvifPassword = onvifPassword; }

    public String getRtspUrl(boolean highRes) {
        String stream = highRes ? "stream1" : "stream2";
        // Return URL without credentials, we will pass them as options for better reliability
        return String.format("rtsp://%s:%d/%s", ip, rtspPort, stream);
    }
}
