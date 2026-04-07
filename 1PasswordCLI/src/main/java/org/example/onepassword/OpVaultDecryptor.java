package org.example.onepassword;

import org.example.onepassword.utils.ByteUtils;

import javax.crypto.Cipher;
import javax.crypto.Mac;
import javax.crypto.SecretKeyFactory;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.PBEKeySpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;

public class OpVaultDecryptor {
    //https://support.1password.com/cs/opvault-design/

    public static final String OP_DATA_MAGIC = "opdata01";

    /**
     * Derives the 64-byte master unlock key from the password using PBKDF2-HMAC-SHA512.
     */
    public static byte[] deriveKeys(String password, String saltBase64, int iterations) throws Exception {
        byte[] salt = Base64.getDecoder().decode(saltBase64);
        
        byte[] passwordBytes = password.getBytes(StandardCharsets.UTF_8);
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
            throw new Exception("decryptOpData - Data too short for opdata01: " + data.length);
        }
        
        // OPVault opdata01 key split: Encryption key first, MAC key second.
        byte[] encKey = Arrays.copyOfRange(derivedKey, 0, 32);
        byte[] macKey = Arrays.copyOfRange(derivedKey, 32, 64);

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[8];
        buffer.get(magic);
        String magicStr = new String(magic, StandardCharsets.US_ASCII);
        if (!magicStr.equals(OP_DATA_MAGIC)) {
            throw new Exception("decryptOpData - Invalid magic: " + ByteUtils.bytesToHex(magic) + " (" + magicStr + ")");
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
        
        if (!MessageDigest.isEqual(expectedMac, computedMac)) {
            throw new Exception("decryptOpData - HMAC mismatch");
        }

        // 2. Decrypt AES-256-CBC
        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(encKey, "AES"), new IvParameterSpec(iv));
        byte[] decrypted = cipher.doFinal(ciphertext);

        // Extract the original plaintext (last plainLen bytes)
        int decryptedLen = decrypted.length;
        if (plainLen > decryptedLen) {
            throw new Exception("decryptOpData - plaintext length in header (" + plainLen + ") is greater than decrypted buffer length (" + decryptedLen + ")");
        }

        return Arrays.copyOfRange(decrypted, decryptedLen - (int)plainLen, decryptedLen);
    }

    public static String decryptItems(byte[] itemKeyData, byte[] detailData, byte[] vaultMasterKey) throws Exception {
        if (itemKeyData.length < 48) {
            throw new Exception("decryptItems - itemKeyData too short: " + itemKeyData.length);
        }

        byte[] vaultEncKey = Arrays.copyOfRange(vaultMasterKey, 0, 32);
        byte[] vaultMacKey = Arrays.copyOfRange(vaultMasterKey, 32, 64);

        // itemKeyData structure: [IV (16 bytes)][Ciphertext (N bytes)][HMAC (32 bytes)]
        int hmacOffset = itemKeyData.length - 32;
        byte[] iv = Arrays.copyOfRange(itemKeyData, 0, 16);
        byte[] ciphertext = Arrays.copyOfRange(itemKeyData, 16, hmacOffset);
        byte[] storedMac = Arrays.copyOfRange(itemKeyData, hmacOffset, itemKeyData.length);

        // 1. Verify HMAC-SHA256 (over IV + Ciphertext)
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(vaultMacKey, "HmacSHA256"));
        mac.update(itemKeyData, 0, hmacOffset);
        byte[] computedMac = mac.doFinal();

        if (!MessageDigest.isEqual(computedMac, storedMac)) {
            throw new Exception("decryptItems - HMAC mismatch");
        }

        // 2. Decrypt Ciphertext with vaultEncKey
        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(vaultEncKey, "AES"), new IvParameterSpec(iv));
        byte[] itemKey = cipher.doFinal(ciphertext);

        // 3. Decrypt Detail Data with the derived itemKey
        // itemKey should be 64 bytes (32 enc, 32 mac) for decryptOpData
        if (itemKey.length < 64) {
            throw new Exception("decryptItems - itemKey too short: " + itemKey.length);
            // If it's shorter, maybe it needs hashing? But OPVault standard says it's 64 bytes.
            // Some older versions might be different, but let's try the direct 64 bytes first.
        }

        return new String(decryptOpData(detailData, itemKey), StandardCharsets.UTF_8);
    }
}
