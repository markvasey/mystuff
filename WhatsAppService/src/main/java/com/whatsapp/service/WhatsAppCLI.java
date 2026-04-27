package com.whatsapp.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import it.auties.whatsapp.api.Whatsapp;
import it.auties.whatsapp.model.jid.Jid;
import it.auties.whatsapp.api.QrHandler;
import it.auties.whatsapp.api.WebHistorySetting;
import it.auties.whatsapp.api.ErrorHandler;
import it.auties.whatsapp.controller.ControllerSerializer;
import it.auties.whatsapp.model.info.ChatMessageInfo;

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

            var builder = Whatsapp.webBuilder()
                    .serializer(serializer);
            
            var connectionOptions = builder.lastOptionalConnection()
                    .orElseGet(builder::firstConnection);

            var api = connectionOptions
                    // Use STANDARD sync instead of DISCARD to ensure initialization
                    .historySetting(WebHistorySetting.standard(false))
                    .errorHandler((client, location, err) -> {
                        if(err != null && err.getMessage() != null && err.getMessage().contains("protocolType")) {
                            System.err.println("[DEBUG] Suppressing protocolType sync error...");
                            return ErrorHandler.Result.DISCARD;
                        }
                        return ErrorHandler.Result.DISCARD;
                    })
                    .unregistered(qr -> {
                        System.out.println("\n" + "=".repeat(50));
                        System.out.println("NEW REGISTRATION REQUIRED! Scan this QR code:");
                        QrHandler.toTerminal().accept(qr);
                        System.out.println("=".repeat(50));
                    })
                    .addLoggedInListener(a -> {
                        System.out.println("\n[EVENT] Successfully Logged In!");
                        loginFuture.complete(null);
                    })
                    .addDisconnectedListener(reason -> {
                        System.out.println("[EVENT] Disconnected: " + reason);
                    })
                    .connect()
                    .join();

            // LOG OUTBOUND NODES
            api.addNodeSentListener(node -> {
                System.out.printf("[NODE SENT] %s (id: %s, to: %s)%n", 
                    node.description(), node.attributes().getString("id"), node.attributes().getString("to"));
            });

            // LOG INBOUND NODES (Extended)
            api.addNodeReceivedListener(node -> {
                String desc = node.description();
                String type = node.attributes().getString("type", "none");
                if (desc.equals("ack") || desc.equals("receipt") || desc.equals("error") || type.equals("error")) {
                    System.out.printf("[NODE RCVD] %s type:%s (id: %s, from: %s)%n", 
                        desc, type, node.attributes().getString("id"), node.attributes().getString("from"));
                }
            });

            api.addMessageStatusListener(info -> {
                if (info instanceof ChatMessageInfo chatInfo) {
                    System.out.printf("[STATUS] Message %s to %s: %s%n", 
                        chatInfo.id(), chatInfo.chatJid(), chatInfo.status());
                }
            });

            if (api.store().jid().isEmpty()) {
                System.out.println("Waiting for handshake/login...");
                try {
                    loginFuture.get(2, TimeUnit.MINUTES);
                } catch (Exception e) {
                    System.err.println("Handshake failed or timed out.");
                }
            }

            // EXTENDED stabilization wait
            System.out.println("Stabilizing connection (30s wait for E2EE sync)...");
            Thread.sleep(30000);

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
                    
                    // Use sendChatMessage for standard text
                    api.sendChatMessage(jid, message)
                       .thenAccept(info -> {
                           System.out.println(">>> Request finished. ID: " + info.id());
                       })
                       .exceptionally(err -> {
                           System.err.println("Failed to send: " + err.getMessage());
                           return null;
                       })
                       .join();

                    Thread.sleep(3000);

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
