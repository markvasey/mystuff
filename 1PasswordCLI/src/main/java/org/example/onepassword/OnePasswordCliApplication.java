package org.example.onepassword;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.Console;
import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.Map;
import java.util.Scanner;

@SpringBootApplication
public class OnePasswordCliApplication implements CommandLineRunner {

    private final ObjectMapper objectMapper = new ObjectMapper();

    public static void main(String[] args) {
        SpringApplication.run(OnePasswordCliApplication.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        System.out.println("--- 1Password OPVault Reader ---");

        String vaultPathStr = "1Password.opvault";
        if (args.length > 0) {
            vaultPathStr = args[0];
        }

        Path vaultPath = Paths.get(vaultPathStr);
        if (!Files.exists(vaultPath)) {
            vaultPath = Paths.get("1PasswordCLI", vaultPathStr);
        }

        if (!Files.exists(vaultPath)) {
            System.err.println("Vault not found at: " + vaultPath.toAbsolutePath());
            return;
        }

        Path profilePath = vaultPath.resolve("default/profile.js");
        if (!Files.exists(profilePath)) {
            System.err.println("Profile not found at: " + profilePath.toAbsolutePath());
            return;
        }

        String password;
        Console console = System.console();
        if (console != null) {
            char[] passwordChars = console.readPassword("Enter Master Password: ");
            password = new String(passwordChars);
        } else {
            System.out.print("Enter Master Password: ");
            Scanner scanner = new Scanner(System.in);
            password = scanner.nextLine();
        }

        try {
            String profileJs = Files.readString(profilePath, StandardCharsets.UTF_8);
            int startIdx = profileJs.indexOf("{");
            int endIdx = profileJs.lastIndexOf("}");
            if (startIdx == -1 || endIdx == -1) {
                System.err.println("Invalid profile.js format");
                return;
            }
            String profileJson = profileJs.substring(startIdx, endIdx + 1);
            VaultProfile profile = objectMapper.readValue(profileJson, VaultProfile.class);

            System.out.println("Deriving keys...");
            byte[] masterUnlockKey = OpVaultDecryptor.deriveKeys(password, profile.getSalt(), profile.getIterations());

            byte[] masterKeyData = Base64.getDecoder().decode(profile.getMasterKey());
            byte[] overviewKeyData = Base64.getDecoder().decode(profile.getOverviewKey());

            byte[] vaultMasterKeyRaw = OpVaultDecryptor.decryptOpData(masterKeyData, masterUnlockKey);
            byte[] vaultOverviewKeyRaw = OpVaultDecryptor.decryptOpData(overviewKeyData, masterUnlockKey);

            java.security.MessageDigest sha512 = java.security.MessageDigest.getInstance("SHA-512");
            byte[] vaultMasterKey = sha512.digest(vaultMasterKeyRaw);
            byte[] vaultOverviewKey = sha512.digest(vaultOverviewKeyRaw);

            System.out.println("Vault unlocked successfully!\n");

            File defaultDir = vaultPath.resolve("default").toFile();
            File[] bandFiles = defaultDir.listFiles((dir, name) -> name.startsWith("band_") && name.endsWith(".js"));

            if (bandFiles != null) {
                int countTitles = 0;
                int countFields = 0;
                for (File bandFile : bandFiles) {
                    if (bandFile.getName().contains("conflicted copy")) continue;
                    
                    try {
                        String bandJs = Files.readString(bandFile.toPath(), StandardCharsets.UTF_8);
                        int start = bandJs.indexOf("{");
                        int end = bandJs.lastIndexOf("}");
                        if (start == -1 || end == -1) continue;
                        
                        String bandJson = bandJs.substring(start, end + 1);
                        Map<String, VaultItem> items = objectMapper.readValue(bandJson, new TypeReference<Map<String, VaultItem>>() {});

                        for (VaultItem item : items.values()) {
                            try {
                                if (item.getO() == null) continue;

                                // 1. Decrypt Overview
                                byte[] overviewData = Base64.getDecoder().decode(item.getO());
                                byte[] decryptedOverview = OpVaultDecryptor.decryptOpData(overviewData, vaultOverviewKey);
                                ItemOverview overview = objectMapper.readValue(decryptedOverview, ItemOverview.class);
                                String title = overview.getTitle() != null ? overview.getTitle() : "No Title";

                                System.out.println("Title: " + title);
                                countTitles++;

                                // 2. Decrypt Details
                                if (item.getD() != null) {
                                    byte[] itemKey = vaultMasterKey;
                                    if (item.getK() != null) {
                                        byte[] encryptedK = Base64.getDecoder().decode(item.getK());
                                        byte[] itemKeyRaw;
                                        
                                        // Check if k is an opdata01 blob
                                        if (encryptedK.length >= 8 && new String(encryptedK, 0, 8, StandardCharsets.US_ASCII).equals(OpVaultDecryptor.OP_DATA_MAGIC)) {
                                            itemKeyRaw = OpVaultDecryptor.decryptOpData(encryptedK, vaultMasterKey);
                                        } else {
                                            // Fallback for 1Password 7 AES-GCM format
                                            itemKeyRaw = OpVaultDecryptor.decryptAesGcm(encryptedK, vaultMasterKey);
                                        }
                                        
                                        // The result of k decryption must be hashed with SHA-512 to get the 64-byte item key
                                        itemKey = sha512.digest(itemKeyRaw);
                                    }

                                    byte[] detailsData = Base64.getDecoder().decode(item.getD());
                                    byte[] decryptedDetails = OpVaultDecryptor.decryptOpData(detailsData, itemKey);
                                    ItemDetails details = objectMapper.readValue(decryptedDetails, ItemDetails.class);

                                    if (details.getFields() != null) {
                                        for (ItemField field : details.getFields()) {
                                            String val = field.getValue() != null ? field.getValue() : "N/A";
                                            String name = field.getName() != null ? field.getName() : (field.getTitle() != null ? field.getTitle() : "Unknown");
                                            System.out.println("  " + name + " : " + val);
                                            countFields++;
                                        }
                                    }
                                }
                                System.out.println();
                            } catch (Exception e) {
                                System.out.println("Failed to decrypt item " + item.getUuid() + ": " + e.getMessage());
                            }
                        }
                    } catch (Exception e) {
                        // Skip unparseable band files
                    }
                }
                System.out.println("Totals Titles: " + countTitles + ", Fields: " + countFields);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
