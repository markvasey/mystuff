package com.whatsapp.service;

import it.auties.whatsapp.api.Whatsapp;
import it.auties.whatsapp.api.ErrorHandler;
import it.auties.whatsapp.api.WebHistorySetting;
import it.auties.whatsapp.api.WebOptionsBuilder;
import it.auties.whatsapp.controller.ControllerSerializer;
import it.auties.whatsapp.model.jid.Jid;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.Scanner;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;
import java.util.Optional;

public class SessionPersistenceTest {

    private static final String TARGET_PHONE = "447557986809"; 

    @Test
    public void testResilientSending() throws Exception {
        // HYPOTHESIS: Dropbox file locking is corrupting the session.
        // We will try to copy the existing session to a local temporary folder.
        Path dropboxSessions = Path.of("sessions");
        Path localSessions = Paths.get(System.getProperty("user.home"), ".whatsapp_test_sessions");
        
        System.out.println("[Test] Dropbox Path: " + dropboxSessions.toAbsolutePath());
        System.out.println("[Test] Local Path: " + localSessions.toAbsolutePath());

        if (Files.exists(dropboxSessions) && !Files.exists(localSessions)) {
            System.out.println("[Test] Initializing local session copy...");
            Files.createDirectories(localSessions);
            // Simple recursive copy (logic omitted for brevity, assuming manual move or just starting fresh local)
        }

        String sessionName = findMostRecentSession(dropboxSessions);
        if (sessionName == null) {
            System.err.println("[Test] No session found. Please run main app once.");
            return;
        }

        var serializer = ControllerSerializer.toProtobuf(dropboxSessions);
        var builder = Whatsapp.webBuilder().serializer(serializer);
        
        Optional<WebOptionsBuilder> optional;
        try {
            UUID uuid = UUID.fromString(sessionName);
            optional = builder.newOptionalConnection(uuid);
        } catch (IllegalArgumentException e) {
            optional = builder.newOptionalConnection(sessionName);
        }
        
        var api = optional.get()
                .historySetting(WebHistorySetting.discard(true))
                .errorHandler((client, location, err) -> {
                    System.err.printf("[WA EVENT] %s: %s%n", location, err != null ? err.getMessage() : "null");
                    return ErrorHandler.Result.DISCARD;
                })
                .registered() 
                .orElseThrow();

        System.out.println("[Test] Connecting...");
        api.connect().get(2, TimeUnit.MINUTES);
        System.out.println("[Test] Handshake complete.");

        // Monitor traffic
        api.addNodeSentListener(node -> {
            if (node.description().equals("message")) {
                System.out.println("[TRAFFIC] >>> OUTBOUND MESSAGE NODE SENT: " + node.attributes().getString("id"));
            }
        });

        System.out.println("[Test] Stabilizing (10s)...");
        Thread.sleep(10000);

        Scanner scanner = new Scanner(System.in);
        int count = 1;
        while (true) {
            System.out.print("\nType message #" + count + " (or 'exit'): ");
            if (!scanner.hasNextLine()) break;
            String text = scanner.nextLine();
            if (text.equalsIgnoreCase("exit")) break;

            System.out.println("[Test] Timestamp: " + java.time.LocalTime.now());
            System.out.println("[Test] Calling api.sendMessage...");
            
            try {
                // We send and DON'T block the main thread with get() immediately
                // to see if the library background threads are working.
                var jid = Jid.of(TARGET_PHONE + "@s.whatsapp.net");
                
                api.sendMessage(jid, text)
                   .thenAccept(info -> System.out.println("[Test] SUCCESS: Acknowledged by server. ID: " + info.id()))
                   .exceptionally(err -> {
                       System.err.println("[Test] ERROR: " + err.getMessage());
                       return null;
                   });

                System.out.println("[Test] Call returned. Waiting 5s for delivery logs...");
                Thread.sleep(5000);
                count++;
            } catch (Exception e) {
                System.err.println("[Test] CRASH: " + e.getMessage());
            }
        }

        api.disconnect().join();
    }

    private String findMostRecentSession(Path sessionsPath) {
        Path webPath = sessionsPath.resolve("web");
        if (!Files.exists(webPath)) return null;
        try (Stream<Path> list = Files.list(webPath)) {
            return list.filter(Files::isDirectory)
                       .max(Comparator.comparingLong(p -> p.toFile().lastModified()))
                       .map(p -> p.getFileName().toString())
                       .orElse(null);
        } catch (Exception e) {
            return null;
        }
    }
}
