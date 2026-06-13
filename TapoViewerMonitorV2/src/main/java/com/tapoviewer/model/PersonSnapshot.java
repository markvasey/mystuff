package com.tapoviewer.model;

import java.awt.image.BufferedImage;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public record PersonSnapshot(BufferedImage image, LocalDateTime timestamp, String cameraName) {
    private static final DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    @Override
    public String toString() {
        return (cameraName != null ? cameraName + " - " : "") + timestamp.format(formatter);
    }
}
