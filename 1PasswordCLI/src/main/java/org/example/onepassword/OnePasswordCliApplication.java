package org.example.onepassword;

import org.example.onepassword.dataClasses.*;
import org.example.onepassword.processors.VaultBandsProcessor;
import org.example.onepassword.processors.VaultKeysProcessor;
import org.example.onepassword.processors.VaultProfileProcessor;
import org.example.onepassword.utils.VaultPathUtils;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.Console;
import java.nio.file.Path;
import java.util.Scanner;

@SpringBootApplication
public class OnePasswordCliApplication implements CommandLineRunner {

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
            byte[] vaultOverviewKey = VaultKeysProcessor.getVaultOverviewKey();

            boolean showRawDecrypted = false;
            boolean filterTrash = true;
            Scanner searchScanner = new Scanner(System.in);

            boolean configureOptions = true;
            while (true) {
                if (configureOptions) {
                    while (true) {
                        System.out.printf("Options: [1] Show raw decrypted: %s, [2] Filter Trash: %s%n",
                                showRawDecrypted ? "Yes" : "No",
                                filterTrash ? "Yes" : "No");
                        System.out.print("Enter 1 or 2 to toggle options, or press Enter to search: ");
                        if (!searchScanner.hasNextLine()) {
                            return;
                        }
                        String optionInput = searchScanner.nextLine().trim();
                        if (optionInput.isEmpty()) {
                            break;
                        }
                        if (optionInput.equalsIgnoreCase("/exit") || optionInput.equalsIgnoreCase("/quit")) {
                            System.out.println("Exiting.");
                            return;
                        }
                        if (optionInput.contains("1")) {
                            showRawDecrypted = !showRawDecrypted;
                        }
                        if (optionInput.contains("2")) {
                            filterTrash = !filterTrash;
                        }
                    }
                    configureOptions = false;
                }

                System.out.println();
                System.out.print("Enter search string (blank for all, /options to change options, /exit or /quit to exit): ");
                if (!searchScanner.hasNextLine()) {
                    break;
                }
                String searchString = searchScanner.nextLine();
                String trimmed = searchString.trim();

                if (trimmed.equalsIgnoreCase("/exit") || trimmed.equalsIgnoreCase("/quit")) {
                    System.out.println("Exiting.");
                    break;
                }
                if (trimmed.equalsIgnoreCase("/options")) {
                    configureOptions = true;
                    continue;
                }

                //Process Bands
                VaultBandsProcessor.ProcessVault(vaultPath, vaultMasterKey, vaultOverviewKey, searchString, showRawDecrypted, filterTrash);
            }
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
