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
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;
import java.util.HexFormat;

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
            throw new Exception("decryptOpData - laintext length in header (" + plainLen + ") is greater than decrypted buffer length (" + decryptedLen + ")");
        }
        
        byte[] returnData = Arrays.copyOfRange(decrypted, decryptedLen - (int)plainLen, decryptedLen);
        //byte[] returnData = Arrays.copyOf(decrypted, (int)plainLen);

    /*
        if(returnData.length != 64) {
            throw new Exception("decryptOpData - returned data length exceed 64");
        }
     */

        return returnData;
    }

    public static byte[] decryptOpData2(byte[] data, byte[] derivedKey) throws Exception {
        if (data.length < 64) {
            throw new Exception("decryptOpData2 - Data too short for opdata01: " + data.length);
        }

        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[8];
        buffer.get(magic);
        String magicStr = new String(magic, StandardCharsets.US_ASCII);
        if (!magicStr.equals(OP_DATA_MAGIC)) {
            throw new Exception("decryptOpData2 - Invalid magic: " + bytesToHex(magic) + " (" + magicStr + ")");
        }

        long plainLen = buffer.getLong();
        byte[] iv = new byte[16];
        buffer.get(iv);

        int hmacLen = 32;
        int cipherLen = data.length - 32 - hmacLen;

        byte[] ciphertext = new byte[cipherLen];
        buffer.get(ciphertext);

        // OPVault opdata01 key split: Encryption key first, MAC key second.
        byte[] encKey = Arrays.copyOfRange(derivedKey, 0, 32);
        byte[] macKey = Arrays.copyOfRange(derivedKey, 32, 64);

        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(macKey, "HmacSHA256"));
        byte[] computed = mac.doFinal(Arrays.copyOfRange(data, 32, data.length));
        byte[] stored   = Arrays.copyOfRange(data, 0, 32);

        if (!MessageDigest.isEqual(stored, computed)) {
            throw new Exception("decryptOpData2 - HMAC mismatch");
        }

        // 2. Decrypt AES-256-CBC
        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding");
        cipher.init(Cipher.DECRYPT_MODE, new SecretKeySpec(encKey, "AES"), new IvParameterSpec(iv));
        byte[] decrypted = cipher.doFinal(ciphertext);

        // Extract the original plaintext (last plainLen bytes)
        int decryptedLen = decrypted.length;
        if (plainLen > decryptedLen) {
            throw new Exception("decryptOpData2 - plaintext length in header (" + plainLen + ") is greater than decrypted buffer length (" + decryptedLen + ")");
        }

        return Arrays.copyOfRange(decrypted, decryptedLen - (int)plainLen, decryptedLen);
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

    public static String decryptItems(byte[] itemKeyData, byte[] detailData, byte[] vaultMasterKey) throws Exception {

        byte[] vaultEncKey = Arrays.copyOfRange(vaultMasterKey, 0, 32);
        byte[] vaultMacKey = Arrays.copyOfRange(vaultMasterKey, 32, 64);

        /*
            Data: 64 bytes
                typedef struct {
                  uint8_t crypto_key[32];
                  uint8_t mac_key[32];
                };
            IV: The data before the MAC is the AES-CBC encrypted item keys using unique random 16-byte IV. - 16 bytes
            MAC: The last 32 bytes comprise the HMAC-SHA256 of the IV and the encrypted data. The MAC is computed with the master MAC key. - 32 bytes
         */

        System.out.println("itemKeyData length:" + itemKeyData.length);

        byte[] stored = Arrays.copyOfRange(itemKeyData, 80,111);
        //byte[] stored = Arrays.copyOfRange(itemKeyData, 0,31);
        //byte[] stored = Arrays.copyOfRange(itemKeyData, 16,47);

        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(vaultMacKey, "HmacSHA256"));
        byte[] computed = mac.doFinal(Arrays.copyOfRange(itemKeyData, 0, 79));
        //byte[] computed = mac.doFinal(Arrays.copyOfRange(itemKeyData, 32, 111));
        //byte[] computed = mac.doFinal(Arrays.copyOfRange(itemKeyData, 48, 111));

        if (!MessageDigest.isEqual(computed, stored)) {
            System.out.println("stored: " + HexFormat.of().formatHex(stored));
            System.out.println("computed: " + HexFormat.of().formatHex(computed));
            throw new Exception("decryptItems - HMAC mismatch");
        }

        byte[] iv = Arrays.copyOfRange(itemKeyData, 32, 48);
        byte[] ciphertext = Arrays.copyOfRange(itemKeyData, 48, itemKeyData.length);

        // 2. Decrypt with vaultKey
        Cipher cipher = Cipher.getInstance("AES/CBC/NoPadding"); // exact 32 bytes, no padding needed
        cipher.init(Cipher.DECRYPT_MODE,
                new SecretKeySpec(vaultEncKey, "AES"),
                new IvParameterSpec(iv));
        byte[] itemKeyRaw = cipher.doFinal(ciphertext);

        MessageDigest sha512 = MessageDigest.getInstance("SHA-512");
        byte[] itemKey = sha512.digest(itemKeyRaw);

        // Result is a UTF-8 JSON string with the item fields
        return new String(decryptOpData(detailData, itemKey), StandardCharsets.UTF_8);
    }
}
