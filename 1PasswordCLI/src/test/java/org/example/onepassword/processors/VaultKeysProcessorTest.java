package org.example.onepassword.processors;

import org.example.onepassword.OpVaultDecryptor;
import org.example.onepassword.dataClasses.VaultProfile;
import org.junit.jupiter.api.Test;

import javax.crypto.Cipher;
import javax.crypto.Mac;
import javax.crypto.spec.IvParameterSpec;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;

import static org.junit.jupiter.api.Assertions.*;

class VaultKeysProcessorTest {

    @Test
    void testUnlockKeysRealistic() throws Exception {
        String password = "master-password";
        String saltBase64 = Base64.getEncoder().encodeToString("salt12345678".getBytes());
        int iterations = 1000;

        // 1. Derive the expected Master Unlock Key (MUK)
        byte[] muk = OpVaultDecryptor.deriveKeys(password, saltBase64, iterations);

        // 2. Prepare raw keys that we expect to get back (before SHA-512)
        byte[] rawMasterKey = new byte[32];
        Arrays.fill(rawMasterKey, (byte) 0xAA);
        byte[] rawOverviewKey = new byte[32];
        Arrays.fill(rawOverviewKey, (byte) 0xBB);

        // 3. Encrypt these keys using opdata01 format with the MUK
        String encryptedMasterKey = Base64.getEncoder().encodeToString(encryptOpData(rawMasterKey, muk));
        String encryptedOverviewKey = Base64.getEncoder().encodeToString(encryptOpData(rawOverviewKey, muk));

        // 4. Create the VaultProfile
        VaultProfile profile = new VaultProfile();
        profile.setSalt(saltBase64);
        profile.setIterations(iterations);
        profile.setMasterKey(encryptedMasterKey);
        profile.setOverviewKey(encryptedOverviewKey);
        profile.setProfileName("TestProfile");

        // 5. Run the processor
        VaultKeysProcessor.UnlockKeys(password, profile);

        // 6. Verify the final keys (should be SHA-512 of raw keys)
        MessageDigest sha512 = MessageDigest.getInstance("SHA-512");
        byte[] expectedMasterKey = sha512.digest(rawMasterKey);
        byte[] expectedOverviewKey = sha512.digest(rawOverviewKey);

        assertArrayEquals(expectedMasterKey, VaultKeysProcessor.getVaultMasterKey());
        assertArrayEquals(expectedOverviewKey, VaultKeysProcessor.getVaultOverviewKey());
    }

    @Test
    void testUnlockKeysWrongPassword() throws Exception {
        String password = "master-password";
        String saltBase64 = Base64.getEncoder().encodeToString("salt12345678".getBytes());
        int iterations = 1000;

        byte[] muk = OpVaultDecryptor.deriveKeys(password, saltBase64, iterations);
        byte[] rawKey = new byte[32];
        String encryptedKey = Base64.getEncoder().encodeToString(encryptOpData(rawKey, muk));

        VaultProfile profile = new VaultProfile();
        profile.setSalt(saltBase64);
        profile.setIterations(iterations);
        profile.setMasterKey(encryptedKey);
        profile.setOverviewKey(encryptedKey);

        assertThrows(RuntimeException.class, () -> VaultKeysProcessor.UnlockKeys("wrong-password", profile));
    }

    private byte[] encryptOpData(byte[] plainBytes, byte[] keys) throws Exception {
        byte[] encKey = Arrays.copyOfRange(keys, 0, 32);
        byte[] macKey = Arrays.copyOfRange(keys, 32, 64);
        byte[] iv = new byte[16];
        Arrays.fill(iv, (byte) 0x01);

        // Pad to 16 bytes
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
}
