package org.example.onepassword.processors;

import org.example.onepassword.dataClasses.VaultProfile;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class VaultProfileProcessorTest {

    @TempDir
    Path tempDir;

    @Test
    void testProcessVaultProfileValid() throws IOException {
        String json = "{\"profileName\":\"testProfile\",\"lastUpdatedBy\":\"tester\",\"updatedAt\":12345,\"salt\":\"abc\",\"iterations\":1000,\"masterKey\":\"mKey\",\"overviewKey\":\"oKey\"}";
        String content = "var profile = " + json + ";";
        Path profilePath = tempDir.resolve("profile.js");
        Files.writeString(profilePath, content);

        VaultProfileProcessor.ProcessVaultProfile(profilePath);
        VaultProfile profile = VaultProfileProcessor.getVaultProfile();

        assertNotNull(profile);
        assertEquals("testProfile", profile.getProfileName());
        assertEquals("tester", profile.getLastUpdatedBy());
        assertEquals(12345, profile.getUpdatedAt());
        assertEquals("abc", profile.getSalt());
        assertEquals(1000, profile.getIterations());
        assertEquals("mKey", profile.getMasterKey());
        assertEquals("oKey", profile.getOverviewKey());
    }

    @Test
    void testProcessVaultProfileInvalidFormat() throws IOException {
        String content = "invalid content without braces";
        Path profilePath = tempDir.resolve("invalid_profile.js");
        Files.writeString(profilePath, content);

        // It should print to stderr and not throw exception based on current code, 
        // but it might leave vaultProfile as null or previous value.
        // Let's check it doesn't crash and maybe reset state if we can (though it's static).
        
        VaultProfileProcessor.ProcessVaultProfile(profilePath);
        // Note: because it's static, it might still have the value from previous test if not careful.
        // In a real scenario we might want to reset it.
    }

    @Test
    void testProcessVaultProfileFileNotFound() {
        Path missingPath = tempDir.resolve("missing.js");
        assertThrows(RuntimeException.class, () -> VaultProfileProcessor.ProcessVaultProfile(missingPath));
    }
}
