package com.whatsapp.service;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.*;

public class WhatsAppClientTest {

    @TempDir
    Path tempDir;

    @Test
    public void testInitializeCreatesDirectory() throws IOException {
        Path sessionsPath = tempDir.resolve("sessions");
        WhatsAppClient client = new WhatsAppClient(sessionsPath);
        
        assertFalse(Files.exists(sessionsPath));
        client.initialize();
        assertTrue(Files.exists(sessionsPath));
    }

    @Test
    public void testAwaitHandshakeWithoutStartThrowsIllegalState() {
        WhatsAppClient client = new WhatsAppClient(tempDir);
        assertThrows(IllegalStateException.class, () -> client.awaitHandshake(1, TimeUnit.SECONDS));
    }

    @Test
    public void testAwaitLoginWithoutHandshakeThrowsIllegalState() {
        WhatsAppClient client = new WhatsAppClient(tempDir);
        assertThrows(IllegalStateException.class, () -> client.awaitLogin(1, TimeUnit.SECONDS));
    }

    @Test
    public void testSendMessageFailsWhenNotConnected() {
        WhatsAppClient client = new WhatsAppClient(tempDir);
        var future = client.sendMessage("447000000000", "Hello");
        assertTrue(future.isCompletedExceptionally());
    }
}
