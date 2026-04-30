package com.whatsapp.service;

import it.auties.whatsapp.api.*;
import it.auties.whatsapp.controller.ControllerSerializer;
import it.auties.whatsapp.model.info.MessageInfo;
import it.auties.whatsapp.model.jid.Jid;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class WhatsAppClient {
    private Whatsapp api;
    private final Path sessionsPath;
    private final CompletableFuture<Void> loginFuture = new CompletableFuture<>();
    private CompletableFuture<Whatsapp> apiFuture;

    public WhatsAppClient(Path sessionsPath) {
        this.sessionsPath = sessionsPath;
    }

    public void initialize() throws IOException {
        if (!Files.exists(sessionsPath)) {
            Files.createDirectories(sessionsPath);
        }
    }

    public WebOptionsBuilder createConnectionOptions(String sessionName) {
        var serializer = ControllerSerializer.toProtobuf(sessionsPath);
        var builder = Whatsapp.webBuilder().serializer(serializer);
        var optional = builder.newOptionalConnection(sessionName);
        
        if (optional.isPresent()) {
            System.out.println("[Client] Existing session found. Resuming...");
            return optional.get();
        } else {
            System.out.println("[Client] No session found. Starting new connection...");
            return builder.firstConnection();
        }
    }

    public void start(WebOptionsBuilder options, String phone) {
        System.out.println("[Client] Initializing connection for " + phone + "...");
        
        // We will use the QR code as the primary method because it's more reliable,
        // but we'll also print any pairing codes we receive.
        this.apiFuture = options
                .historySetting(WebHistorySetting.discard(true))
                .errorHandler(createErrorHandler())
                .unregistered(qr -> {
                    System.out.println("\n[Client] QR CODE RECEIVED! Scan this with WhatsApp:");
                    QrHandler.toTerminal().accept(qr);
                    System.out.println("[Client] Alternatively, if you see an 8-character code below, you can use that.");
                })
                .addLoggedInListener(a -> {
                    System.out.println("[Client] Login event detected!");
                    loginFuture.complete(null);
                })
                .connect();
    }

    public void awaitHandshake(long timeout, TimeUnit unit) throws Exception {
        if (apiFuture == null) throw new IllegalStateException("Connection not started");
        try {
            System.out.println("[Client] Waiting for handshake (Linking)...");
            this.api = apiFuture.get(timeout, unit);
            System.out.println("[Client] Handshake successful.");
        } catch (TimeoutException e) {
            System.err.println("[Client] ERROR: Handshake timed out. Did you scan the QR code?");
            throw e;
        }
    }

    public void awaitLogin(long timeout, TimeUnit unit) throws Exception {
        if (api == null) throw new IllegalStateException("Handshake not complete");
        if (api.store().jid().isEmpty()) {
            System.out.println("[Client] Waiting for final login confirmation...");
            loginFuture.get(timeout, unit);
        } else {
            System.out.println("[Client] Logged in as: " + api.store().jid().get());
        }
    }

    private ErrorHandler createErrorHandler() {
        return (client, location, err) -> {
            if (err != null && err.getMessage() != null && err.getMessage().contains("protocolType")) {
                // Silently drop the known sync error
                return ErrorHandler.Result.DISCARD;
            }
            // For first connection errors, sometimes DISCARD is safer to avoid loops
            return ErrorHandler.Result.DISCARD;
        };
    }

    public CompletableFuture<? extends MessageInfo<?>> sendMessage(String phone, String text) {
        if (api == null || !api.isConnected()) {
            return CompletableFuture.failedFuture(new IllegalStateException("Not connected"));
        }
        return api.sendMessage(Jid.of(phone + "@s.whatsapp.net"), text);
    }

    public void disconnect() {
        if (api != null) {
            System.out.println("[Client] Disconnecting...");
            api.disconnect().join();
        }
    }

    public boolean isReady() {
        return api != null && api.isConnected() && api.store().jid().isPresent();
    }
}
