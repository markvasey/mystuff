package org.example.onepassword.processors;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.OpVaultDecryptor;
import org.example.onepassword.dataClasses.ItemDetails;
import org.example.onepassword.dataClasses.ItemField;
import org.example.onepassword.dataClasses.ItemOverview;
import org.example.onepassword.dataClasses.VaultItem;

import java.util.Base64;
import java.util.Map;

public class DisplayProcessor {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    private static int countTitles;
    private static int countFields;

    static {
        countTitles = 0;
        countFields = 0;
    }

    public static int getCountTitles() {
        return countTitles;
    }

    public static int getCountFields() {
        return countFields;
    }

    public static void resetCounters() {
        countTitles = 0;
        countFields = 0;
    }

    public static void DisplayResults(Map<String, VaultItem> items, byte[] vaultMasterKey, byte[] vaultOverviewKey, String searchString) {

        int itemCounter = 0;
        for (VaultItem item : items.values()) {
            try {
                if (item.getO() == null) continue;

                // 1. Decrypt Overview
                byte[] overviewData = Base64.getDecoder().decode(item.getO());
                byte[] decryptedOverview = OpVaultDecryptor.decryptOpData(overviewData, vaultOverviewKey);
                ItemOverview overview = objectMapper.readValue(decryptedOverview, ItemOverview.class);
                String title = overview.getTitle() != null ? overview.getTitle() : "No Title";

                // Filter by search string
                if (searchString != null && !searchString.isBlank() &&
                        !title.toLowerCase().contains(searchString.toLowerCase())) {
                        continue;
                }

                itemCounter++;

                System.out.println("VaultItem: " + itemCounter);
                System.out.println();

                String categoryNameText = item.getCategory() == null ? "" : item.getCategory() + " - ";
                
                String urlText = overview.getUrl() == null ? "" : ", URL: " + overview.getUrl();

                System.out.println(categoryNameText + "Title: " + title + urlText);
                countTitles++;

                // 2. Decrypt Details
                if (item.getD() != null && item.getK() != null) {
                    byte[] encryptedK = Base64.getDecoder().decode(item.getK());
                    byte[] encryptedD = Base64.getDecoder().decode(item.getD());

                    try {
                        String decryptedDetailsJson = OpVaultDecryptor.decryptItems(encryptedK, encryptedD, vaultMasterKey);
                        System.out.println("Decrypted: " + decryptedDetailsJson);
                        ItemDetails details = objectMapper.readValue(decryptedDetailsJson, ItemDetails.class);

                        if (details.getFields() != null) {
                            for (ItemField field : details.getFields()) {
                                String idText = field.getId() == null ? "" : "[Id: " + field.getId() + "] ";
                                System.out.print(idText + field.getName() + " = " + field.getValue());

                                String type = field.getType() == null ? "" : "Type: " + field.getType();
                                String designation = field.getDesignation() == null ? "" : "Designation: " + field.getDesignation();
                                if (!type.isEmpty() || !designation.isEmpty()) {
                                    System.out.print(" (");
                                    if (!type.isEmpty()) {
                                        System.out.print(type);
                                    }
                                    if (!type.isEmpty() && !designation.isEmpty()) {
                                        System.out.print(", ");
                                    }
                                    if (!designation.isEmpty()) {
                                        System.out.print(designation);
                                    }
                                    System.out.println(")");
                                }

                                countFields++;
                            }
                        }
                        System.out.println();
                        if (details.getPassword() != null && !details.getPassword().isEmpty()) {
                            System.out.println("P assword: " + details.getPassword());
                        }
                        if (details.getNumber() != null && !details.getNumber().isEmpty()) {
                            System.out.println(" Number: " + details.getNumber());
                        }
                        if (details.getMembership_no() != null && !details.getMembership_no().isEmpty()) {
                            System.out.println(" Membership No.: " + details.getMembership_no());
                        }
                        if (details.getNotesPlain() != null && !details.getNotesPlain().isEmpty()) {
                            System.out.println(" Notes: " + details.getNotesPlain());
                        }
                    } catch (Exception e) {
                        System.out.println("  Failed to decrypt details: " + e.getMessage());
                    }
                }
                System.out.println();
                System.out.println("------------------------------------------------------------------");
                System.out.println();

            } catch (Exception e) {
                System.out.println("Failed to read item " + item.getUuid() + ": " + e.getMessage());
            }
        }

        System.out.println("Totals Titles: " + countTitles + ", Fields: " + countFields);

    }
}
