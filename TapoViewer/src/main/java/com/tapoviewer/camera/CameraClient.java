package com.tapoviewer.camera;

import com.tapoviewer.model.CameraSettings;
import io.github.hyeonmo.client.OnvifClient;
import io.github.hyeonmo.models.OnvifDevice;
import io.github.hyeonmo.models.OnvifMediaProfile;
import io.github.hyeonmo.models.ptz.PtzType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;

public class CameraClient {
    private static final Logger logger = LoggerFactory.getLogger(CameraClient.class);
    private final CameraSettings settings;
    private OnvifDevice onvifDevice;

    public CameraClient(CameraSettings settings) {
        this.settings = settings;
    }

    public CompletableFuture<Void> connect() {
        String onvifUrl = settings.getIp() + ":" + settings.getOnvifPort();
        return OnvifClient.connect(onvifUrl)
                .credentials(settings.getOnvifUsername(), settings.getOnvifPassword())
                .buildAsync()
                .thenAccept(device -> {
                    this.onvifDevice = device;
                    logger.info("Connected to camera via ONVIF: {}", settings.getIp());
                })
                .thenCompose(v -> onvifDevice.media().getMediaProfiles())
                .thenAccept(profiles -> {
                    if (!profiles.isEmpty()) {
                        logger.info("Retrieved {} ONVIF profiles.", profiles.size());
                    } else {
                        logger.warn("No ONVIF profiles found.");
                    }
                });
    }

    public CompletableFuture<Void> ptzMove(PtzType type) {
        if (onvifDevice == null) {
            return CompletableFuture.failedFuture(new IllegalStateException("Not connected to ONVIF."));
        }
        return onvifDevice.ptz().move(type)
                .thenCompose(v -> {
                    // Stop movement after a short delay for "tap" behavior
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    return onvifDevice.ptz().stop();
                }).thenAccept(v -> {});
    }

    public CompletableFuture<Void> ptzStop() {
        if (onvifDevice == null) {
            return CompletableFuture.failedFuture(new IllegalStateException("Not connected to ONVIF."));
        }
        return onvifDevice.ptz().stop().thenAccept(v -> {});
    }
}
