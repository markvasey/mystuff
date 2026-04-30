package com.whatsapp.service;

import java.util.concurrent.TimeUnit;

public class WhatsAppOrchestrator {
    private final WhatsAppClient client;
    private final AccountManager accountManager;

    public WhatsAppOrchestrator(WhatsAppClient client, AccountManager accountManager) {
        this.client = client;
        this.accountManager = accountManager;
    }

    public void startService(String sessionName) throws Exception {
        client.initialize();
        
        var accounts = accountManager.getAccounts();
        if (accounts.isEmpty()) {
            throw new IllegalStateException("No accounts available");
        }

        var options = client.createConnectionOptions(sessionName);
        client.start(options, accounts.get(0).phone());
        
        System.out.println("[Orchestrator] Handshake started...");
        client.awaitHandshake(10, TimeUnit.MINUTES);
        
        System.out.println("[Orchestrator] Waiting for login...");
        client.awaitLogin(2, TimeUnit.MINUTES);
        
        System.out.println("[Orchestrator] Service Ready.");
    }

    public void send(int accountIndex, String message) throws Exception {
        var account = accountManager.getAccount(accountIndex);
        if (account == null) {
            throw new IllegalArgumentException("Invalid account index");
        }

        client.sendMessage(account.phone(), message)
                .thenAccept(info -> System.out.println(">>> Sent: " + info.id()))
                .get(20, TimeUnit.SECONDS);
    }

    public void stop() {
        client.disconnect();
    }
}
