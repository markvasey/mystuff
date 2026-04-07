package org.example.onepassword.utils;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class VaultPathUtils {

    public static Path getVaultPath(String... args) {
        String vaultPathStr = "1Password.opvault";
        if (args.length > 0) {
            vaultPathStr = args[0];
        }

        Path vaultPath = Paths.get(vaultPathStr);
        if (!Files.exists(vaultPath)) {
            vaultPath = Paths.get("1PasswordCLI", vaultPathStr);
        }

        return vaultPath;
    }

    public static Path getProfilePath(Path vaultPath) {
        Path profilePath = vaultPath.resolve("default/profile.js");
        if (!Files.exists(profilePath)) {
            System.err.println("Profile not found at: " + profilePath.toAbsolutePath());
            return null;
        }
        return profilePath;
    }
}
