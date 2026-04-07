package org.example.onepassword.processors;

import org.example.onepassword.OpVaultDecryptor;
import org.example.onepassword.dataClasses.VaultItem;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.Mac;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;

class DisplayProcessorTest {

    @Test
    void testDisplayResultsRealistic() throws Exception {
        byte[] vaultMasterKey = new byte[64];
        Arrays.fill(vaultMasterKey, (byte) 0x11);
        byte[] vaultOverviewKey = new byte[64];
        Arrays.fill(vaultOverviewKey, (byte) 0x22);

        Map<String, VaultItem> items = new HashMap<>();

        // 1. Login item
        items.put("UUID1", createMockItem("001", "My Login", "https://site.com", 
            "{\"fields\":[{\"name\":\"username\",\"value\":\"user1\",\"type\":\"T\"},{\"name\":\"password\",\"value\":\"pass1\",\"type\":\"P\",\"designation\":\"password\"}]}", 
            vaultMasterKey, vaultOverviewKey));

        // 2. Credit Card
        items.put("UUID2", createMockItem("002", "My Visa", null, 
            "{\"fields\":[{\"name\":\"cardholder\",\"value\":\"Mark\"}], \"number\":\"1234-5678\"}", 
            vaultMasterKey, vaultOverviewKey));

        // 3. Secure Note
        items.put("UUID3", createMockItem("003", "Top Secret", null, 
            "{\"notesPlain\":\"Keep it secret, keep it safe.\"}", 
            vaultMasterKey, vaultOverviewKey));

        // 4. Membership
        items.put("UUID4", createMockItem("105", "Costco", null, 
            "{\"membership_no\":\"999888\"}", 
            vaultMasterKey, vaultOverviewKey));

        // 5. Item with null Overview (should be skipped)
        VaultItem nullO = new VaultItem();
        nullO.setUuid("UUID5");
        items.put("UUID5", nullO);

        // 6. Item with invalid decryption (should be caught and logged)
        VaultItem invalidD = createMockItem("001", "Invalid Item", null, "{}", vaultMasterKey, vaultOverviewKey);
        invalidD.setD("not-base64-garbage!!!");
        items.put("UUID6", invalidD);

        assertDoesNotThrow(() -> DisplayProcessor.DisplayResults(items, vaultMasterKey, vaultOverviewKey, null));
        
        // Additional test for search filtering
        assertDoesNotThrow(() -> DisplayProcessor.DisplayResults(items, vaultMasterKey, vaultOverviewKey, "Visa"));
    }

    private VaultItem createMockItem(String category, String title, String url, String detailsJson, byte[] vMasterKey, byte[] vOverviewKey) throws Exception {
        // 1. Overview 'o'
        String overviewJson = "{\"title\":\"" + title + "\"" + (url != null ? ",\"url\":\"" + url + "\"" : "") + "}";
        String encryptedO = Base64.getEncoder().encodeToString(encryptOpData(overviewJson, vOverviewKey));

        // 2. Item Key 'k'
        byte[] itemKeys = new byte[64];
        Arrays.fill(itemKeys, (byte) 0x33);
        String encryptedK = Base64.getEncoder().encodeToString(encryptItemKey(itemKeys, vMasterKey));

        // 3. Details 'd'
        String encryptedD = Base64.getEncoder().encodeToString(encryptOpData(detailsJson, itemKeys));

        VaultItem item = new VaultItem();
        item.setCategory(category);
        item.setO(encryptedO);
        item.setK(encryptedK);
        item.setD(encryptedD);
        item.setUuid("MOCK-" + title);
        return item;
    }

    private byte[] encryptOpData(String json, byte[] keys) throws Exception {
        byte[] plainBytes = json.getBytes(StandardCharsets.UTF_8);
        byte[] encKey = Arrays.copyOfRange(keys, 0, 32);
        byte[] macKey = Arrays.copyOfRange(keys, 32, 64);
        byte[] iv = new byte[16];
        Arrays.fill(iv, (byte) 0x01);

        int paddedLen = ((plainBytes.length + 15) / 16) * 16;
        byte[] paddedPlaintext = new byte[paddedLen];
        System.arraycopy(plainBytes, 0, paddedPlaintext, paddedLen - plainBytes.length, plainBytes.length);

        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(encKey, "AES"), new IvParameterSpec(iv));
        byte[] ciphertext = cipher.doFinal(paddedPlaintext);

        int totalLen = 8 + 8 + 16 + ciphertext.length + 32;
        ByteBuffer buffer = ByteBuffer.allocate(totalLen).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("opdata01".getBytes(StandardCharsets.US_ASCII));
        buffer.putLong(plainBytes.length);
        buffer.put(iv);
        buffer.put(ciphertext);

        Mac hmac = Mac.getInstance("HmacSHA256");
        hmac.init(new SecretKeySpec(macKey, "HmacSHA256"));
        hmac.update(buffer.array(), 0, totalLen - 32);
        buffer.put(hmac.doFinal());

        return buffer.array();
    }

    private byte[] encryptItemKey(byte[] itemKeys, byte[] vMasterKey) throws Exception {
        byte[] vEncKey = Arrays.copyOfRange(vMasterKey, 0, 32);
        byte[] vMacKey = Arrays.copyOfRange(vMasterKey, 32, 64);
        byte[] iv = new byte[16];
        Arrays.fill(iv, (byte) 0x02);

        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(vEncKey, "AES"), new IvParameterSpec(iv));
        byte[] ciphertext = cipher.doFinal(itemKeys);

        int totalLen = 16 + ciphertext.length + 32;
        ByteBuffer buffer = ByteBuffer.allocate(totalLen);
        buffer.put(iv);
        buffer.put(ciphertext);

        Mac hmac = Mac.getInstance("HmacSHA256");
        hmac.init(new SecretKeySpec(vMacKey, "HmacSHA256"));
        hmac.update(buffer.array(), 0, totalLen - 32);
        buffer.put(hmac.doFinal());

        return buffer.array();
    }
}
