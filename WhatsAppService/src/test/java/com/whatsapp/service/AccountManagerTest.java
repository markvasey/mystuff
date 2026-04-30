package com.whatsapp.service;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

public class AccountManagerTest {

    @Test
    public void testLoadValidJson() throws Exception {
        String json = """
        {
          "accounts": [
            {"name": "Test", "phone": "12345"}
          ]
        }
        """;
        InputStream is = new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8));
        AccountManager manager = new AccountManager();
        
        manager.load(is);
        
        assertEquals(1, manager.getAccounts().size());
        assertEquals("Test", manager.getAccount(0).name());
        assertEquals("12345", manager.getAccount(0).phone());
    }

    @Test
    public void testLoadEmptyJson() throws Exception {
        String json = "{\"accounts\": []}";
        InputStream is = new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8));
        AccountManager manager = new AccountManager();
        
        manager.load(is);
        
        assertTrue(manager.getAccounts().isEmpty());
    }

    @Test
    public void testGetAccountOutOfBounds() {
        AccountManager manager = new AccountManager();
        assertNull(manager.getAccount(0));
        assertNull(manager.getAccount(-1));
    }
}
