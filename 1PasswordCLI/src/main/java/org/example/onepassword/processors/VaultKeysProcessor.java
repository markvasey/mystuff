package org.example.onepassword.processors;

import org.example.onepassword.OpVaultDecryptor;
import org.example.onepassword.dataClasses.VaultProfile;

import java.util.Base64;

public class VaultKeysProcessor {
    private static byte[] vaultMasterKey;
    private static byte[] vaultOverviewKey;

    public static void UnlockKeys(String password, VaultProfile profile) {
        System.out.println("Deriving keys...");

        try {
            byte[] masterUnlockKey = OpVaultDecryptor.deriveKeys(password, profile.getSalt(), profile.getIterations());

            byte[] masterKeyData = Base64.getDecoder().decode(profile.getMasterKey());
            byte[] overviewKeyData = Base64.getDecoder().decode(profile.getOverviewKey());

            byte[] vaultMasterKeyRaw = OpVaultDecryptor.decryptOpData(masterKeyData, masterUnlockKey);
            byte[] vaultOverviewKeyRaw = OpVaultDecryptor.decryptOpData(overviewKeyData, masterUnlockKey);

            java.security.MessageDigest sha512 = java.security.MessageDigest.getInstance("SHA-512");
            vaultMasterKey = sha512.digest(vaultMasterKeyRaw);
            vaultOverviewKey = sha512.digest(vaultOverviewKeyRaw);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        System.out.println("Vault unlocked successfully!\n");
    }

    public static byte[] getVaultMasterKey() {
        return vaultMasterKey;
    }

    public static byte[] getVaultOverviewKey() {
        return vaultOverviewKey;
    }
}
