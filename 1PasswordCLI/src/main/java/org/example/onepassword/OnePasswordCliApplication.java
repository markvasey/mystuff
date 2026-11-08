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

            String[][] categoryMenu = {
                    {"0", "ALL", "All items"},
                    {"1", "001", "Logins"},
                    {"2", "003", "Secure Notes"},
                    {"3", "002", "Credit Cards"},
                    {"4", "005", "Passwords"},
                    {"5", "109", "Wireless Routers"},
                    {"6", "106", "Passports"},
                    {"7", "107", "Reward Programmes"},
                    {"8", "105", "Memberships"},
                    {"9", "103", "Driver Licenses"},
                    {"10", "ARCHIVE", "Archive"}
            };

            boolean configureOptions = true;
            while (true) {
                if (configureOptions) {
                    while (true) {
                        System.out.printf("Options: [1] Show raw decrypted: %s, [2] Filter Trash: %s%n",
                                showRawDecrypted ? "Yes" : "No",
                                filterTrash ? "Yes" : "No");
                        System.out.print("Enter 1 or 2 to toggle options, or press Enter to continue: ");
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

                java.util.Map<String, VaultItem> allItems = VaultBandsProcessor.loadMergedItems(vaultPath);
                boolean currentFilterTrash = filterTrash;
                long allCount = allItems.values().stream()
                        .filter(item -> !currentFilterTrash || !Boolean.TRUE.equals(item.getTrashed()))
                        .count();
                long archiveCount = allItems.values().stream()
                        .filter(item -> Boolean.TRUE.equals(item.getTrashed()))
                        .count();

                System.out.println();
                System.out.println("Categories:");
                for (String[] entry : categoryMenu) {
                    String num = entry[0];
                    String code = entry[1];
                    String label = entry[2];
                    long count;
                    if ("ALL".equals(code)) {
                        count = allCount;
                    } else if ("ARCHIVE".equals(code)) {
                        count = archiveCount;
                    } else {
                        count = allItems.values().stream()
                                .filter(item -> !currentFilterTrash || !Boolean.TRUE.equals(item.getTrashed()))
                                .filter(item -> code.equals(item.getCategoryCode()))
                                .count();
                    }
                    String defaultSuffix = "0".equals(num) ? " [default]" : "";
                    System.out.printf("  [%s] %s (%d)%s%n", num, label, count, defaultSuffix);
                }

                System.out.print("Select category [0-10, Enter for All items, /options, /exit]: ");
                if (!searchScanner.hasNextLine()) {
                    break;
                }
                String catInput = searchScanner.nextLine().trim();
                if (catInput.equalsIgnoreCase("/exit") || catInput.equalsIgnoreCase("/quit")) {
                    System.out.println("Exiting.");
                    break;
                }
                if (catInput.equalsIgnoreCase("/options")) {
                    configureOptions = true;
                    continue;
                }

                String selectedCategoryCode = "ALL";
                String selectedCategoryLabel = "All items";
                if (!catInput.isEmpty()) {
                    for (String[] entry : categoryMenu) {
                        if (entry[0].equals(catInput)
                                || entry[1].equalsIgnoreCase(catInput)
                                || entry[2].equalsIgnoreCase(catInput)) {
                            selectedCategoryCode = entry[1];
                            selectedCategoryLabel = entry[2];
                            break;
                        }
                    }
                }

                System.out.printf("Enter search string for %s (blank for all, /options, /exit or /quit): ", selectedCategoryLabel);
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
                VaultBandsProcessor.ProcessVault(vaultPath, vaultMasterKey, vaultOverviewKey, searchString, showRawDecrypted, filterTrash, selectedCategoryCode);
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
