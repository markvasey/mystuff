package com.whatsapp.service;

import java.nio.file.Path;
import java.util.Scanner;

public class WhatsAppCLI {
    private static final String SESSION_NAME = "MainCLI";

    public static void main(String[] args) {
        AccountManager accountManager = new AccountManager();
        WhatsAppClient client = new WhatsAppClient(Path.of("sessions"));
        WhatsAppOrchestrator orchestrator = new WhatsAppOrchestrator(client, accountManager);

        try {
            // Load accounts
            try (var is = WhatsAppCLI.class.getClassLoader().getResourceAsStream("accounts.json")) {
                accountManager.load(is);
            }

            // Start Service
            orchestrator.startService(SESSION_NAME);

            // UI Loop
            Scanner scanner = new Scanner(System.in);
            while (true) {
                var accounts = accountManager.getAccounts();
                System.out.println("\n--- WhatsApp CLI ---");
                for (int i = 0; i < accounts.size(); i++) {
                    System.out.printf("%d. %s (%s)%n", i + 1, accounts.get(i).name(), accounts.get(i).phone());
                }
                System.out.println("X. Exit");
                System.out.print("Choice: ");
                
                String choice = scanner.nextLine().trim();
                if (choice.equalsIgnoreCase("X")) break;

                try {
                    int selection = Integer.parseInt(choice) - 1;
                    System.out.print("Message: ");
                    String message = scanner.nextLine();
                    orchestrator.send(selection, message);
                } catch (Exception e) {
                    System.err.println("Error: " + e.getMessage());
                }
            }

        } catch (Exception e) {
            System.err.println("FATAL ORCHESTRATION ERROR");
            e.printStackTrace();
        } finally {
            orchestrator.stop();
        }
    }
}
