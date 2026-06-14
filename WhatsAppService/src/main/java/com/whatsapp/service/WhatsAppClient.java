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
            System.out.println("[Client] Resuming session: " + sessionName);
            return optional.get();
        } else {
            System.out.println("[Client] Initializing new session...");
            return builder.newConnection(sessionName);
        }
    }

    public void start(WebOptionsBuilder options, String phone) {
        this.apiFuture = options
                .errorHandler(createErrorHandler())
                .unregistered(qr -> {
                    System.out.println("\n[Client] QR CODE REQUIRED. Please scan:");
                    QrHandler.toTerminal().accept(qr);
                })
                .addLoggedInListener(a -> {
                    System.out.println("[Client] Session authenticated.");
                    loginFuture.complete(null);
                })
                .connect()
                .thenApply(whatsapp -> {
                    whatsapp.addDisconnectedListener(reason -> {
                        System.err.println("[Client WA Event] Socket disconnected. Reason: " + reason);
                    });
                    return whatsapp;
                });
    }

    public void awaitHandshake(long timeout, TimeUnit unit) throws Exception {
        if (apiFuture == null) throw new IllegalStateException("Not started");
        try {
            this.api = apiFuture.get(timeout, unit);
            System.out.println("[Client] Handshake finished.");
        } catch (TimeoutException e) {
            System.err.println("[Client] Handshake timeout.");
            throw e;
        }
    }

    public void awaitLogin(long timeout, TimeUnit unit) throws Exception {
        if (api == null) throw new IllegalStateException("No api instance");
        if (api.store().jid().isEmpty()) {
            System.out.println("[Client] Waiting for final login event...");
            loginFuture.get(timeout, unit);
        }
    }

    private ErrorHandler createErrorHandler() {
        return (client, location, err) -> {
            System.err.printf("[Client WA Event] Error at %s: %s%n", location, err != null ? err.getMessage() : "null");
            if (location == ErrorHandler.Location.STREAM || location == ErrorHandler.Location.LOGIN) {
                return ErrorHandler.Result.RECONNECT; // Recover on connection or login drops/timeouts
            }
            return ErrorHandler.Result.DISCARD; // Discard non-fatal message parsing or state errors
        };
    }

    public boolean isConnected() {
        return api != null && api.isConnected();
    }

    public CompletableFuture<? extends MessageInfo<?>> sendMessage(String phone, String text) {
        if (api == null || !api.isConnected()) {
            return CompletableFuture.failedFuture(new IllegalStateException("Not connected"));
        }
        
        // Wake up session - observed as helpful in unit tests
        api.changePresence(true);
        
        return api.sendMessage(Jid.of(phone + "@s.whatsapp.net"), text);
    }

    public void disconnect() {
        if (api != null) {
            System.out.println("[Client] Disconnecting...");
            api.disconnect().join();
        }
    }
}
