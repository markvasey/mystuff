package org.example.onepassword;

import javax.crypto.Cipher;
import javax.crypto.Mac;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Base64;

public class OpVaultDecryptor {
    public static final String OP_DATA_MAGIC = "opdata01";

    /**
     * Derives the 64-byte master unlock key from the password using PBKDF2-HMAC-SHA512.
     */
    public static byte[] deriveKeys(String password, String saltBase64, int iterations) throws Exception {
        byte[] salt = Base64.getDecoder().decode(saltBase64);
        
        byte[] passwordBytes = password.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        char[] passwordChars = new char[passwordBytes.length];
        for (int i = 0; i < passwordBytes.length; i++) {
            passwordChars[i] = (char) (passwordBytes[i] & 0xFF);
        }

        PBEKeySpec spec = new PBEKeySpec(passwordChars, salt, iterations, 512);
        SecretKeyFactory skf = SecretKeyFactory.getInstance("PBKDF2WithHmacSHA512");
        return skf.generateSecret(spec).getEncoded();
    }

    /**
     * Decrypts data in the 'opdata01' format (AES-256-CBC + HMAC-SHA256).
     */
    public static byte[] decryptOpData(byte[] data, byte[] derivedKey) throws Exception {
        if (data.length < 64) {
            throw new Exception("Data too short for opdata01: " + data.length);
        }
        
        // OPVault opdata01 key split: Encryption key first, MAC key second.
        byte[] encKey = Arrays.copyOfRange(derivedKey, 0, 32);
        byte[] macKey = Arrays.copyOfRange(derivedKey, 32, 64);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[8];
        buffer.get(magic);
        String magicStr = new String(magic, java.nio.charset.StandardCharsets.US_ASCII);
        if (!magicStr.equals(OP_DATA_MAGIC)) {
            throw new Exception("decryptOpData - Invalid magic: " + bytesToHex(magic) + " (" + magicStr + ")");
        }

        long plainLen = buffer.getLong();
        byte[] iv = new byte[16];
        buffer.get(iv);

        int hmacLen = 32;
        int cipherLen = data.length - 32 - hmacLen;
        
        byte[] ciphertext = new byte[cipherLen];
        buffer.get(ciphertext);
        byte[] expectedMac = new byte[hmacLen];
        buffer.get(expectedMac);

        // 1. Verify HMAC-SHA256
        Mac hmac = Mac.getInstance("HmacSHA256");
        hmac.init(new SecretKeySpec(macKey, "HmacSHA256"));
        hmac.update(data, 0, data.length - hmacLen);
        byte[] computedMac = hmac.doFinal();
        
        if (!java.security.MessageDigest.isEqual(expectedMac, computedMac)) {
            throw new Exception("decryptOpData - HMAC mismatch");
        }

        // 2. Decrypt AES-256-CBC
        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(encKey, "AES"), new IvParameterSpec(iv));
        byte[] decrypted = cipher.doFinal(ciphertext);

        // Extract the original plaintext (last plainLen bytes)
        int decryptedLen = decrypted.length;
        if (plainLen > decryptedLen) {
            throw new Exception("PdecryptOpData - laintext length in header (" + plainLen + ") is greater than decrypted buffer length (" + decryptedLen + ")");
        }
        
        return Arrays.copyOfRange(decrypted, decryptedLen - (int)plainLen, decryptedLen);
    }

    /**
     * Decrypts data using AES-256-GCM (used for some 1Password 7 items).
     * Structure: [IV(12)][Ciphertext(n)][Tag(16)]
     */
    public static byte[] decryptAesGcm(byte[] data, byte[] key) throws Exception {
        if (data.length < 28) {
            throw new Exception("decryptAesGcm - Data too short for AES-GCM: " + data.length);
        }

        byte[] iv = Arrays.copyOfRange(data, 0, 12);
        byte[] ciphertextWithTag = Arrays.copyOfRange(data, 12, data.length);

        Cipher cipher = Cipher.getInstance("AES/GCM/NoPadding");
        // 1Password uses 128-bit (16-byte) authentication tag for GCM
        GCMParameterSpec spec = new GCMParameterSpec(128, iv);
        
        // Ensure we only use the first 32 bytes of the key for AES-256
        byte[] aesKey = key.length > 32 ? Arrays.copyOfRange(key, 0, 32) : key;
        
        cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(aesKey, "AES"), spec);
        return cipher.doFinal(ciphertextWithTag);
    }

    private static final char[] HEX_ARRAY = "0123456789ABCDEF".toCharArray();
    public static String bytesToHex(byte[] bytes) {
        char[] hexChars = new char[bytes.length * 2];
        for (int j = 0; j < bytes.length; j++) {
            int v = bytes[j] & 0xFF;
            hexChars[j * 2] = HEX_ARRAY[v >>> 4];
            hexChars[j * 2 + 1] = HEX_ARRAY[v & 0x0F];
        }
        return new String(hexChars);
    }
}
