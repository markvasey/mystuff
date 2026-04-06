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
            https://support.1password.com/cs/opvault-design/

            Data: 64 bytes
                typedef struct {
                  uint8_t crypto_key[32];
                  uint8_t mac_key[32];
                };
            IV: The data before the MAC is the AES-CBC encrypted item keys using unique random 16-byte IV. - 16 bytes
            MAC: The last 32 bytes comprise the HMAC-SHA256 of the IV and the encrypted data. The MAC is computed with the master MAC key. - 32 bytes
         */
        /*
            https://darthnull.org/1pass-local-vaults/

            The key_data structure includes four components:

                Initialization Vector (IV) (16 bytes)
                Item Encryption Key (32 bytes)
                Item HMAC Key (32 bytes)
                HMAC Tag (32 bytes)
            First, compute the HMAC tag using the encrypted item keys (encryption + HMAC) as the message, and the Master HMAC Key as key.
            If that matches the HMAC tag found in the structure, then we know it hasn’t been altered.
            Now, use the Master AES key to decrypt the item keys, and those keys (AES + HMAC) to decrypt the actual vault item.
         */

        ByteBuffer buffer = ByteBuffer.wrap(itemKeyData).order(ByteOrder.LITTLE_ENDIAN);
        byte[] iv = new byte[16];
        buffer.get(iv);
        byte[] ciphertext = new byte[64];
        buffer.get(ciphertext);
        byte[] stored = new byte[32];
        buffer.get(stored);

        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(vaultMacKey, "HmacSHA256"));

        byte[] computed = mac.doFinal(ciphertext);

        if (!MessageDigest.isEqual(computed, stored)) {
            throw new Exception("decryptItems - HMAC mismatch");
        }

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
