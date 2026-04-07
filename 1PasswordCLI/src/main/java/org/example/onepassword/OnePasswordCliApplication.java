package org.example.onepassword;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.dataClasses.*;
import org.example.onepassword.processors.VaultBandsProcessor;
import org.example.onepassword.processors.VaultKeysProcessor;
import org.example.onepassword.processors.VaultProfileProcessor;
import org.example.onepassword.utils.VaultPathUtils;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.Console;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;
import java.util.Scanner;

@SpringBootApplication
public class OnePasswordCliApplication implements CommandLineRunner {

    private final ObjectMapper objectMapper = new ObjectMapper();

    public static void main(String[] args) {
        SpringApplication.run(OnePasswordCliApplication.class, args);
    }

    @Override
    public void run(String... args) {
        System.out.println("--- 1Password OPVault Reader ---");

        Path vaultPath = VaultPathUtils.getVaultPath(args);
        Path profilePath = VaultPathUtils.getProfilePath(vaultPath);

        String password = PromptForPassword();

        try {
            //Process Profile
            VaultProfileProcessor.ProcessVaultProfile(profilePath);
            VaultProfile profile = VaultProfileProcessor.getVaultProfile();

            //Derive Master Keys
            VaultKeysProcessor.UnlockKeys(password, profile);
            byte[] vaultMasterKey = VaultKeysProcessor.getVaultMasterKey();
            byte[] vaultOverviewKey =  VaultKeysProcessor.getVaultOverviewKey();

            //Process Bands
            VaultBandsProcessor.ProcessVault(vaultPath,vaultMasterKey,vaultOverviewKey);
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
        }
    }

    private String PromptForPassword() {
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
        return password;
    }
}
