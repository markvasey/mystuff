package org.example.onepassword.processors;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.dataClasses.VaultProfile;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public class VaultProfileProcessor {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    private static VaultProfile vaultProfile;

    public static void ProcessVaultProfile(Path profilePath){
        try {
            String profileJs = Files.readString(profilePath, StandardCharsets.UTF_8);
            int startIdx = profileJs.indexOf("{");
            int endIdx = profileJs.lastIndexOf("}");
            if (startIdx == -1 || endIdx == -1) {
                System.err.println("Invalid profile.js format");
                return;
            }
            String profileJson = profileJs.substring(startIdx, endIdx + 1);
            //System.out.println("profileJson: " + profileJson);

            vaultProfile = objectMapper.readValue(profileJson, VaultProfile.class);

            System.out.println("Profile: " + vaultProfile.getProfileName() + ", Updated By: " + vaultProfile.getLastUpdatedBy() + ", Updated At: " + vaultProfile.getUpdatedAt());
        } catch (RuntimeException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static VaultProfile getVaultProfile() {
        return vaultProfile;
    }
}
