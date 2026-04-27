package com.whatsapp.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import it.auties.whatsapp.api.Whatsapp;
import it.auties.whatsapp.model.jid.Jid;
import it.auties.whatsapp.api.QrHandler;
import it.auties.whatsapp.api.WebHistorySetting;
import it.auties.whatsapp.api.ErrorHandler;
import it.auties.whatsapp.controller.ControllerSerializer;

import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class WhatsAppCLI {

    private static class Account {
        String name;
        String phone;

        Account(String name, String phone) {
            this.name = name;
            this.phone = phone;
        }
    }

    public static void main(String[] args) {
        try {
            List<Account> accounts = loadAccounts();
            if (accounts.isEmpty()) {
                System.err.println("No accounts found in accounts.json");
                return;
            }

            Scanner scanner = new Scanner(System.in);
            System.out.println("Starting WhatsApp connection (Cobalt 0.0.10)...");
            
            Path sessionsPath = Path.of("sessions");
            var serializer = ControllerSerializer.toProtobuf(sessionsPath);

            CompletableFuture<Void> loginFuture = new CompletableFuture<>();

            var api = Whatsapp.webBuilder()
                    .serializer(serializer)
                    .firstConnection()
                    .historySetting(WebHistorySetting.discard(false))
                    .errorHandler((client, location, err) -> {
                        if(err != null && err.getMessage() != null && err.getMessage().contains("protocolType")) {
                            // DISCARD is critical here to prevent the loop
                            return ErrorHandler.Result.DISCARD;
                        }
                        // For other errors during setup, just try to keep going
                        return ErrorHandler.Result.DISCARD;
                    })
                    .unregistered(qr -> {
                        System.out.println("\n" + "=".repeat(50));
                        System.out.println("QR CODE RECEIVED!");
                        System.out.println("RAW DATA: " + qr);
                        System.out.println("=".repeat(50));
                        System.out.println("Rendering terminal QR...");
                        try {
                            QrHandler.toTerminal().accept(qr);
                        } catch (Exception e) {
                            System.out.println("Terminal QR Error: " + e.getMessage());
                        }
                    })
                    .addLoggedInListener(a -> {
                        System.out.println("\n[EVENT] Successfully Logged In!");
                        loginFuture.complete(null);
                    })
                    .connect()
                    .join();

            if (api.store().jid().isEmpty()) {
                System.out.println("Waiting for handshake/login... (2 minute timeout)");
                try {
                    loginFuture.get(2, TimeUnit.MINUTES);
                } catch (Exception e) {
                    System.err.println("Handshake failed or timed out. Status: " + e.getMessage());
                }
            }

            // Even if it times out, if JID is present now, we proceed
            if (api.store().jid().isPresent()) {
                System.out.println("Session active for: " + api.store().jid().get());
            } else {
                System.err.println("No active session. Application may fail to send.");
            }

            System.out.println("Stabilizing connection (10s wait)...");
            Thread.sleep(10000);

            while (true) {
                System.out.println("\n--- WhatsApp CLI Service ---");
                for (int i = 0; i < accounts.size(); i++) {
                    System.out.printf("%d. %s (%s)%n", i + 1, accounts.get(i).name, accounts.get(i).phone);
                }
                System.out.println("X. Exit");
                System.out.print("Select an account (number or X): ");
                
                String choice = scanner.nextLine().trim();
                if (choice.equalsIgnoreCase("X")) {
                    break;
                }

                try {
                    int selection = Integer.parseInt(choice) - 1;
                    if (selection < 0 || selection >= accounts.size()) {
                        System.err.println("Invalid selection.");
                        continue;
                    }

                    Account target = accounts.get(selection);
                    System.out.print("Enter message for " + target.name + ": ");
                    String message = scanner.nextLine();

                    System.out.println("Sending message...");
                    Jid jid = Jid.of(target.phone + "@s.whatsapp.net");
                    
                    api.sendMessage(jid, message)
                       .thenAccept(ack -> System.out.println("Message acknowledged by server!"))
                       .exceptionally(err -> {
                           System.err.println("Failed to send: " + err.getMessage());
                           return null;
                       })
                       .join();

                    System.out.println("Waiting 5s for transmission...");
                    Thread.sleep(5000);

                } catch (NumberFormatException e) {
                    System.err.println("Invalid input.");
                }
            }

            System.out.println("Closing connection...");
            api.disconnect().join();
            System.out.println("Done!");
            System.exit(0);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static List<Account> loadAccounts() throws Exception {
        List<Account> list = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream is = WhatsAppCLI.class.getClassLoader().getResourceAsStream("accounts.json")) {
            if (is == null) return list;
            JsonNode root = mapper.readTree(is);
            JsonNode accountsNode = root.get("accounts");
            if (accountsNode.isArray()) {
                for (JsonNode node : accountsNode) {
                    list.add(new Account(node.get("name").asText(), node.get("phone").asText()));
                }
            }
        }
        return list;
    }
}
