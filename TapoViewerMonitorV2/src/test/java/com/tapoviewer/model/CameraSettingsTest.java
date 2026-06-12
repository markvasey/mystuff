package com.tapoviewer.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class CameraSettingsTest {

    @Test
    public void testGetRtspUrl() {
        CameraSettings settings = new CameraSettings();
        settings.setIp("192.168.1.100");
        settings.setRtspPort(554);
        
        // Test high resolution stream
        String urlHigh = settings.getRtspUrl(true);
        assertEquals("rtsp://192.168.1.100:554/stream1", urlHigh);
        
        // Test low resolution stream
        String urlLow = settings.getRtspUrl(false);
        assertEquals("rtsp://192.168.1.100:554/stream2", urlLow);
    }

    @Test
    public void testDefaultValues() {
        CameraSettings settings = new CameraSettings();
        assertEquals("Tapo Camera", settings.getName());
        assertEquals(554, settings.getRtspPort());
        assertEquals(2020, settings.getOnvifPort());
    }
}
