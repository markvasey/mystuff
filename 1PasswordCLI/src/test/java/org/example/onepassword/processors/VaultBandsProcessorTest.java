package org.example.onepassword.processors;

import org.example.onepassword.OpVaultDecryptor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import javax.crypto.Cipher;
import javax.crypto.Mac;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;

import static org.junit.jupiter.api.Assertions.*;

class VaultBandsProcessorTest {

    @TempDir
    Path tempDir;

    @Test
    void testProcessVaultRealistic() throws Exception {
        Path vaultPath = tempDir.resolve("vault");
        Path defaultDir = vaultPath.resolve("default");
        Files.createDirectories(defaultDir);

        byte[] vaultMasterKey = new byte[64];
        Arrays.fill(vaultMasterKey, (byte) 0x11);
        byte[] vaultOverviewKey = new byte[64];
        Arrays.fill(vaultOverviewKey, (byte) 0x22);

        // Band 0: Login items
        String band0 = "ld({" +
                "\"UUID1\": " + createMockItem("Login", "Google", "https://google.com", "{\"fields\":[{\"name\":\"username\",\"value\":\"mark@gmail.com\"},{\"name\":\"password\",\"value\":\"p4ssw0rd\"}]}", vaultMasterKey, vaultOverviewKey) + "," +
                "\"UUID2\": " + createMockItem("Login", "GitHub", "https://github.com", "{\"fields\":[{\"name\":\"username\",\"value\":\"mvasey\"},{\"name\":\"password\",\"value\":\"git-pass\"}]}", vaultMasterKey, vaultOverviewKey) +
                "});";
        Files.writeString(defaultDir.resolve("band_0.js"), band0);

        // Band 1: Secure Notes
        String band1 = "ld({" +
                "\"UUID3\": " + createMockItem("Secure Note", "Secret Code", null, "{\"notesPlain\":\"The code is 42.\"}", vaultMasterKey, vaultOverviewKey) +
                "});";
        Files.writeString(defaultDir.resolve("band_1.js"), band1);

        assertDoesNotThrow(() -> VaultBandsProcessor.ProcessVault(vaultPath, vaultMasterKey, vaultOverviewKey));
    }

    private String createMockItem(String category, String title, String url, String detailsJson, byte[] vMasterKey, byte[] vOverviewKey) throws Exception {
        // 1. Overview 'o'
        String overviewJson = "{\"title\":\"" + title + "\"" + (url != null ? ",\"url\":\"" + url + "\"" : "") + "}";
        String encryptedO = Base64.getEncoder().encodeToString(encryptOpData(overviewJson, vOverviewKey));

        // 2. Item Key 'k'
        byte[] itemKeys = new byte[64];
        Arrays.fill(itemKeys, (byte) 0x33);
        String encryptedK = Base64.getEncoder().encodeToString(encryptItemKey(itemKeys, vMasterKey));

        // 3. Details 'd'
        String encryptedD = Base64.getEncoder().encodeToString(encryptOpData(detailsJson, itemKeys));

        return "{" +
                "\"category\":\"" + category + "\"," +
                "\"o\":\"" + encryptedO + "\"," +
                "\"k\":\"" + encryptedK + "\"," +
                "\"d\":\"" + encryptedD + "\"," +
                "\"uuid\":\"MOCK-UUID\"" +
                "}";
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
