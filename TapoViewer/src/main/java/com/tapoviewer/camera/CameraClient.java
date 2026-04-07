package com.tapoviewer.camera;

import com.tapoviewer.model.CameraSettings;
import io.github.hyeonmo.client.OnvifClient;
import io.github.hyeonmo.models.OnvifDevice;
import io.github.hyeonmo.models.ptz.PtzType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class CameraClient {
    private static final Logger logger = LoggerFactory.getLogger(CameraClient.class);
    private final CameraSettings settings;
    private OnvifDevice onvifDevice;
    private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

    public CameraClient(CameraSettings settings) {
        this.settings = settings;
    }

    public CompletableFuture<Boolean> connect() {
        String onvifUrl = settings.getIp() + ":" + settings.getOnvifPort();
        return OnvifClient.connect(onvifUrl)
                .credentials(settings.getOnvifUsername(), settings.getOnvifPassword())
                .buildAsync()
                .thenAccept(device -> {
                    this.onvifDevice = device;
                    logger.info("Connected to camera via ONVIF: {}", settings.getIp());
                })
                .thenCompose(v -> onvifDevice.device().getCapabilities())
                .thenAccept(caps -> {
                    if (caps.getPtzXaddr() != null) {
                        String ptzPath = caps.getPtzXaddr();
                        if (ptzPath.contains("/onvif/")) {
                            ptzPath = ptzPath.substring(ptzPath.indexOf("/onvif/"));
                        }
                        onvifDevice.getPath().setPtzPath(ptzPath);
                    }
                })
                .thenCompose(v -> onvifDevice.ptz().getStatus())
                .thenApply(status -> {
                    // If getStatus contains "zoom", we assume it's supported.
                    // Many fixed-lens Tapo cameras return a status without a Zoom node.
                    boolean zoomSupported = status != null && status.toLowerCase().contains("zoom");
                    logger.info("Zoom support detected: {}", zoomSupported);
                    return zoomSupported;
                })
                .exceptionally(ex -> {
                    logger.warn("Could not determine zoom support, defaulting to false: {}", ex.getMessage());
                    return false;
                });
    }

    public CompletableFuture<Void> ptzMove(PtzType type) {
        if (onvifDevice == null) {
            return CompletableFuture.failedFuture(new IllegalStateException("Not connected to ONVIF."));
        }
        
        logger.info("Sending PTZ Move: {} to path {}", type, onvifDevice.getPath().getPtzPath());
        
        return onvifDevice.ptz().move(type)
                .thenCompose(v -> {
                    CompletableFuture<Void> stopFuture = new CompletableFuture<>();
                    scheduler.schedule(() -> {
                        onvifDevice.ptz().stop()
                            .thenAccept(res -> stopFuture.complete(null))
                            .exceptionally(ex -> {
                                stopFuture.completeExceptionally(ex);
                                return null;
                            });
                    }, 500, TimeUnit.MILLISECONDS);
                    return stopFuture;
                });
    }

    public CompletableFuture<Void> ptzStop() {
        if (onvifDevice == null) {
            return CompletableFuture.failedFuture(new IllegalStateException("Not connected to ONVIF."));
        }
        return onvifDevice.ptz().stop().thenAccept(v -> {});
    }

    public void release() {
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(1, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
        }
    }
}
