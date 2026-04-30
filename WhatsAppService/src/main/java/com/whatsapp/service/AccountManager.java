package com.whatsapp.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class AccountManager {
    public record Account(String name, String phone) {}

    private final List<Account> accounts = new ArrayList<>();

    public void load(InputStream inputStream) throws Exception {
        accounts.clear();
        if (inputStream == null) {
            return;
        }
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(inputStream);
        JsonNode accountsNode = root.get("accounts");
        if (accountsNode != null && accountsNode.isArray()) {
            for (JsonNode node : accountsNode) {
                accounts.add(new Account(node.get("name").asText(), node.get("phone").asText()));
            }
        }
    }

    public List<Account> getAccounts() {
        return List.copyOf(accounts);
    }

    public Account getAccount(int index) {
        if (index < 0 || index >= accounts.size()) {
            return null;
        }
        return accounts.get(index);
    }
}
