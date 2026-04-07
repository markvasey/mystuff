package org.example.onepassword;

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

        byte[] paddedPlaintext = new byte[16];
        System.arraycopy(plainBytes, 0, paddedPlaintext, 16 - plainBytes.length, plainBytes.length);
        
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
        byte[] macResult = hmac.doFinal();
        buffer.put(macResult);

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
    void testDecryptItemsLogin() throws Exception {
        String detailsJson = "{\"fields\":[{\"name\":\"username\",\"value\":\"markvasey\"},{\"name\":\"password\",\"value\":\"correct-horse-battery-staple\"}]}";
        runDecryptItemsTest(detailsJson);
    }

    @Test
    void testDecryptItemsCreditCard() throws Exception {
        String detailsJson = "{\"fields\":[{\"name\":\"cardholder\",\"value\":\"Mark Vasey\"},{\"name\":\"ccnum\",\"value\":\"1234-5678-9012-3456\"},{\"name\":\"expiry\",\"value\":\"12/2028\"}]}";
        runDecryptItemsTest(detailsJson);
    }

    @Test
    void testDecryptItemsSecureNote() throws Exception {
        String detailsJson = "{\"notesPlain\":\"This is a very secret note that should be decrypted correctly by the program.\"}";
        runDecryptItemsTest(detailsJson);
    }

    @Test
    void testDecryptItemsComplex() throws Exception {
        String detailsJson = "{\"fields\":[{\"name\":\"username\",\"value\":\"mark\"}],\"sections\":[{\"title\":\"Extra\",\"fields\":[{\"t\":\"Pin\",\"v\":\"1234\"}]}],\"notesPlain\":\"Some comments here.\"}";
        runDecryptItemsTest(detailsJson);
    }

    private void runDecryptItemsTest(String detailsJson) throws Exception {
        // 1. Derive Vault Master Key using the actual method
        byte[] vaultMasterKeyFull = OpVaultDecryptor.deriveKeys(testPassword, testSaltBase64, testIterations);
        // vaultMasterKey used in decryptItems is actually the output of SHA-512 of decrypted master key data from profile.js
        // But in our test setup, we just need a 64-byte key. 
        // Let's assume vaultMasterKey passed to decryptItems is already 64 bytes.
        byte[] vaultMasterKey = vaultMasterKeyFull; 

        byte[] vaultEncKey = Arrays.copyOfRange(vaultMasterKey, 0, 32);
        byte[] vaultMacKey = Arrays.copyOfRange(vaultMasterKey, 32, 64);

        // 2. Generate random Item Keys (64 bytes)
        byte[] itemKeys = new byte[64];
        for(int i=0; i<64; i++) itemKeys[i] = (byte)i;
        
        // 3. Encrypt Item Keys for 'k' field: [IV (16)][Ciphertext (64)][HMAC (32)]
        byte[] kIv = new byte[16];
        Arrays.fill(kIv, (byte) 0xAA);
        Cipher kCipher = Cipher.getInstance("AES/CBC/NoPadding");
        kCipher.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(vaultEncKey, "AES"), new IvParameterSpec(kIv));
        byte[] kCiphertext = kCipher.doFinal(itemKeys);
        
        int kLen = 16 + kCiphertext.length + 32;
        ByteBuffer kBuffer = ByteBuffer.allocate(kLen);
        kBuffer.put(kIv);
        kBuffer.put(kCiphertext);
        
        Mac kMac = Mac.getInstance("HmacSHA256");
        kMac.init(new SecretKeySpec(vaultMacKey, "HmacSHA256"));
        kMac.update(kBuffer.array(), 0, kLen - 32);
        kBuffer.put(kMac.doFinal());
        
        byte[] encryptedK = kBuffer.array();

        // 4. Generate Details Data 'd' in opdata01 format
        byte[] detailsBytes = detailsJson.getBytes(StandardCharsets.UTF_8);
        
        // Pad to 16 bytes for CBC
        int paddedLen = ((detailsBytes.length + 15) / 16) * 16;
        byte[] paddedDetails = new byte[paddedLen];
        // 1Password pads with random bytes at the start, but we'll pad at the end for simplicity in test generation
        // Actually, our decryptOpData expects plaintext at the end.
        System.arraycopy(detailsBytes, 0, paddedDetails, paddedLen - detailsBytes.length, detailsBytes.length);
        
        byte[] dIv = new byte[16];
        Arrays.fill(dIv, (byte) 0xBB);
        
        byte[] itemEncKey = Arrays.copyOfRange(itemKeys, 0, 32);
        byte[] itemMacKey = Arrays.copyOfRange(itemKeys, 32, 64);
        
        Cipher dCipher = Cipher.getInstance("AES/CBC/NoPadding");
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

        // 5. Test decryptItems
        String result = OpVaultDecryptor.decryptItems(encryptedK, encryptedD, vaultMasterKey);
        assertEquals(detailsJson, result);
    }
}
