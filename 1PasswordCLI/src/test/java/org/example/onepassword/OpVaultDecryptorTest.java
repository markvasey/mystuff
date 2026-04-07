package org.example.onepassword;

import org.junit.jupiter.api.BeforeEach;
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

import static org.junit.jupiter.api.Assertions.*;

class OpVaultDecryptorTest {

    private String testPassword = "test-password";
    private String testSaltBase64 = "YmFzZTY0LXNhbHQ="; // "base64-salt"
    private int testIterations = 1000;

    @Test
    void testDeriveKeys() throws Exception {
        byte[] keys = OpVaultDecryptor.deriveKeys(testPassword, testSaltBase64, testIterations);
        assertNotNull(keys);
        assertEquals(64, keys.length);
        
        // Verify consistency
        byte[] keys2 = OpVaultDecryptor.deriveKeys(testPassword, testSaltBase64, testIterations);
        assertArrayEquals(keys, keys2);
        
        // Verify different password gives different key
        byte[] keys3 = OpVaultDecryptor.deriveKeys("wrong-password", testSaltBase64, testIterations);
        assertFalse(Arrays.equals(keys, keys3));
    }

    @Test
    void testDecryptOpData() throws Exception {
        byte[] derivedKey = OpVaultDecryptor.deriveKeys(testPassword, testSaltBase64, testIterations);
        byte[] encKey = Arrays.copyOfRange(derivedKey, 0, 32);
        byte[] macKey = Arrays.copyOfRange(derivedKey, 32, 64);

        String plaintext = "Secret Message";
        byte[] plainBytes = plaintext.getBytes(StandardCharsets.UTF_8);
        byte[] iv = new byte[16];
        Arrays.fill(iv, (byte) 0x01);

        // 1. Encrypt data (AES-CBC-NoPadding with manual padding)
        // 1Password uses random padding at the beginning, but NoPadding requires multiple of 16.
        // For simplicity in test, we'll just use 16 bytes of plaintext.
        byte[] paddedPlaintext = new byte[16];
        System.arraycopy(plainBytes, 0, paddedPlaintext, 16 - plainBytes.length, plainBytes.length);
        
        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(encKey, "AES"), new IvParameterSpec(iv));
        byte[] ciphertext = cipher.doFinal(paddedPlaintext);

        // 2. Build opdata01 structure
        // [Magic(8)][Length(8)][IV(16)][Ciphertext(n)][HMAC(32)]
        int totalLen = 8 + 8 + 16 + ciphertext.length + 32;
        ByteBuffer buffer = ByteBuffer.allocate(totalLen).order(ByteOrder.LITTLE_ENDIAN);
        buffer.put("opdata01".getBytes(StandardCharsets.US_ASCII));
        buffer.putLong(plainBytes.length);
        buffer.put(iv);
        buffer.put(ciphertext);
        
        // 3. Calculate HMAC
        Mac hmac = Mac.getInstance("HmacSHA256");
        hmac.init(new SecretKeySpec(macKey, "HmacSHA256"));
        hmac.update(buffer.array(), 0, totalLen - 32);
        byte[] macResult = hmac.doFinal();
        buffer.put(macResult);

        // 4. Decrypt and Verify
        byte[] decrypted = OpVaultDecryptor.decryptOpData(buffer.array(), derivedKey);
        assertEquals(plaintext, new String(decrypted, StandardCharsets.UTF_8));
    }

    @Test
    void testDecryptOpDataInvalidMagic() throws Exception {
        byte[] derivedKey = new byte[64];
        byte[] invalidData = new byte[64];
        System.arraycopy("invalid!".getBytes(), 0, invalidData, 0, 8);
        
        Exception exception = assertThrows(Exception.class, () -> {
            OpVaultDecryptor.decryptOpData(invalidData, derivedKey);
        });
        assertTrue(exception.getMessage().contains("Invalid magic"));
    }

    @Test
    void testDecryptItems() throws Exception {
        // Mock Vault Master Key
        byte[] vaultMasterKey = new byte[64];
        Arrays.fill(vaultMasterKey, (byte) 0x02);
        byte[] vaultEncKey = Arrays.copyOfRange(vaultMasterKey, 0, 32);
        byte[] vaultMacKey = Arrays.copyOfRange(vaultMasterKey, 32, 64);

        // 1. Generate Item Keys (64 bytes)
        byte[] itemKeys = new byte[64];
        Arrays.fill(itemKeys, (byte) 0x03);
        
        // 2. Encrypt Item Keys for 'k' field
        byte[] kIv = new byte[16];
        Arrays.fill(kIv, (byte) 0x04);
        Cipher kCipher = Cipher.getInstance("AES/CBC/NoPadding");
        kCipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(vaultEncKey, "AES"), new IvParameterSpec(kIv));
        byte[] kCiphertext = kCipher.doFinal(itemKeys);
        
        // Build 'k' data: [IV][Ciphertext][HMAC]
        int kLen = 16 + kCiphertext.length + 32;
        ByteBuffer kBuffer = ByteBuffer.allocate(kLen);
        kBuffer.put(kIv);
        kBuffer.put(kCiphertext);
        
        Mac kMac = Mac.getInstance("HmacSHA256");
        kMac.init(new SecretKeySpec(vaultMacKey, "HmacSHA256"));
        kMac.update(kBuffer.array(), 0, kLen - 32);
        kBuffer.put(kMac.doFinal());
        
        byte[] encryptedK = kBuffer.array();

        // 3. Generate Details Data 'd' in opdata01 format
        String detailsJson = "{\"fields\":[{\"name\":\"username\",\"value\":\"user1\"}]}";
        byte[] detailsBytes = detailsJson.getBytes(StandardCharsets.UTF_8);
        
        // Pad to 16 bytes for CBC
        int paddedLen = ((detailsBytes.length + 15) / 16) * 16;
        byte[] paddedDetails = new byte[paddedLen];
        System.arraycopy(detailsBytes, 0, paddedDetails, paddedLen - detailsBytes.length, detailsBytes.length);
        
        byte[] dIv = new byte[16];
        Arrays.fill(dIv, (byte) 0x05);
        
        byte[] itemEncKey = Arrays.copyOfRange(itemKeys, 0, 32);
        byte[] itemMacKey = Arrays.copyOfRange(itemKeys, 32, 64);
        
        Cipher dCipher = Cipher.getInstance("AES/CBC/NoPadding");
        dCipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(itemEncKey, "AES"), new IvParameterSpec(dIv));
        // Wait, I need to encrypt it for the test
        dCipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(itemEncKey, "AES"), new IvParameterSpec(dIv));
        byte[] dCiphertext = dCipher.doFinal(paddedDetails);
        
        int dTotalLen = 8 + 8 + 16 + dCiphertext.length + 32;
        ByteBuffer dBuffer = ByteBuffer.allocate(dTotalLen).order(ByteOrder.LITTLE_ENDIAN);
        dBuffer.put("opdata01".getBytes(StandardCharsets.US_ASCII));
        dBuffer.putLong(detailsBytes.length);
        dBuffer.put(dIv);
        dBuffer.put(dCiphertext);
        
        Mac dMac = Mac.getInstance("HmacSHA256");
        dMac.init(new SecretKeySpec(itemMacKey, "HmacSHA256"));
        dMac.update(dBuffer.array(), 0, dTotalLen - 32);
        dBuffer.put(dMac.doFinal());
        
        byte[] encryptedD = dBuffer.array();

        // 4. Test decryptItems
        String result = OpVaultDecryptor.decryptItems(encryptedK, encryptedD, vaultMasterKey);
        assertEquals(detailsJson, result);
    }
}
