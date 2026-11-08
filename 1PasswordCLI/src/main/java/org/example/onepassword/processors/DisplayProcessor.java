package org.example.onepassword.processors;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.OpVaultDecryptor;
import org.example.onepassword.dataClasses.ItemDetails;
import org.example.onepassword.dataClasses.ItemField;
import org.example.onepassword.dataClasses.ItemOverview;
import org.example.onepassword.dataClasses.ItemSection;
import org.example.onepassword.dataClasses.ItemSectionField;
import org.example.onepassword.dataClasses.OverviewUrl;
import org.example.onepassword.dataClasses.PasswordHistoryEntry;
import org.example.onepassword.dataClasses.VaultItem;
import org.example.onepassword.utils.Utils;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

public class DisplayProcessor {

    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final DateTimeFormatter TIMESTAMP_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

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

    public static void PrintTotals() {
        System.out.println("Totals Titles: " + countTitles + ", Fields: " + countFields);
    }

    public static void DisplayResults(Map<String, VaultItem> items, byte[] vaultMasterKey, byte[] vaultOverviewKey, String searchString) {
        DisplayResults(items, vaultMasterKey, vaultOverviewKey, Utils.parseSearchString(searchString), false, true, "ALL");
    }

    public static void DisplayResults(
            Map<String, VaultItem> items,
            byte[] vaultMasterKey,
            byte[] vaultOverviewKey,
            String searchString,
            boolean showRawDecrypted,
            boolean filterTrash) {
        DisplayResults(items, vaultMasterKey, vaultOverviewKey, Utils.parseSearchString(searchString), showRawDecrypted, filterTrash, "ALL");
    }

    public static void DisplayResults(
            Map<String, VaultItem> items,
            byte[] vaultMasterKey,
            byte[] vaultOverviewKey,
            String searchString,
            boolean showRawDecrypted,
            boolean filterTrash,
            String categoryFilter) {
        DisplayResults(items, vaultMasterKey, vaultOverviewKey, Utils.parseSearchString(searchString), showRawDecrypted, filterTrash, categoryFilter);
    }

    public static void DisplayResults(Map<String, VaultItem> items, byte[] vaultMasterKey, byte[] vaultOverviewKey, List<String> searchTerms) {
        DisplayResults(items, vaultMasterKey, vaultOverviewKey, searchTerms, false, true, "ALL");
    }

    public static void DisplayResults(
            Map<String, VaultItem> items,
            byte[] vaultMasterKey,
            byte[] vaultOverviewKey,
            List<String> searchTerms,
            boolean showRawDecrypted,
            boolean filterTrash) {
        DisplayResults(items, vaultMasterKey, vaultOverviewKey, searchTerms, showRawDecrypted, filterTrash, "ALL");
    }

    public static void DisplayResults(
            Map<String, VaultItem> items,
            byte[] vaultMasterKey,
            byte[] vaultOverviewKey,
            List<String> searchTerms,
            boolean showRawDecrypted,
            boolean filterTrash,
            String categoryFilter) {

        boolean archiveOnly = "ARCHIVE".equalsIgnoreCase(categoryFilter);
        boolean filterBySpecificCategory = categoryFilter != null
                && !categoryFilter.isBlank()
                && !"ALL".equalsIgnoreCase(categoryFilter)
                && !archiveOnly;

        for (VaultItem item : items.values()) {
            try {
                if (archiveOnly) {
                    if (!Boolean.TRUE.equals(item.getTrashed())) continue;
                } else {
                    if (filterTrash && Boolean.TRUE.equals(item.getTrashed())) continue;
                    if (filterBySpecificCategory) {
                        if (!categoryFilter.equals(item.getCategoryCode())
                                && !categoryFilter.equalsIgnoreCase(item.getCategory())) {
                            continue;
                        }
                    }
                }
                if (item.getO() == null) continue;

                // 1. Decrypt Overview
                byte[] overviewData = Base64.getDecoder().decode(item.getO());
                byte[] decryptedOverview = OpVaultDecryptor.decryptOpData(overviewData, vaultOverviewKey);
                ItemOverview overview = objectMapper.readValue(decryptedOverview, ItemOverview.class);
                String title = overview.getTitle() != null ? overview.getTitle() : "No Title";
                String folderPath = VaultFoldersProcessor.getFolderPath(item.getFolder());

                // 2. Decrypt Details (if present) so search and display have access to all item properties
                String decryptedDetailsJson = null;
                ItemDetails details = null;
                Exception detailsError = null;
                if (item.getD() != null && item.getK() != null) {
                    try {
                        byte[] encryptedK = Base64.getDecoder().decode(item.getK());
                        byte[] encryptedD = Base64.getDecoder().decode(item.getD());
                        decryptedDetailsJson = OpVaultDecryptor.decryptItems(encryptedK, encryptedD, vaultMasterKey);
                        details = objectMapper.readValue(decryptedDetailsJson, ItemDetails.class);
                    } catch (Exception e) {
                        detailsError = e;
                    }
                }

                // 3. Filter by search terms (OR logic across title, overview, folder, and details)
                if (searchTerms != null && !searchTerms.isEmpty()) {
                    if (!matchesSearchTerms(item, overview, folderPath, details, searchTerms)) {
                        continue;
                    }
                }

                countTitles++;
                System.out.println("VaultItem: " + countTitles);
                System.out.println();

                String categoryNameText = item.getCategory() == null ? "" : item.getCategory() + " - ";
                String urlText = overview.getUrl() == null ? "" : ", URL: " + overview.getUrl();
                StringBuilder statusFlags = new StringBuilder();
                if (Boolean.TRUE.equals(item.getTrashed())) {
                    statusFlags.append(" [TRASHED]");
                }
                if (item.getFave() != null && item.getFave() > 0) {
                    statusFlags.append(" [FAVORITE]");
                }

                System.out.println(categoryNameText + "Title: " + title + urlText + statusFlags);

                if (showRawDecrypted) {
                    System.out.println("Decrypted Overview: " + new String(decryptedOverview, java.nio.charset.StandardCharsets.UTF_8));
                    if (item.getUuid() != null) {
                        System.out.println("  UUID: " + item.getUuid());
                    }
                }

                // Display Item Envelope & Overview Metadata
                if (folderPath != null && !folderPath.isBlank()) {
                    System.out.println("  Folder: " + folderPath);
                }
                if (overview.getAinfo() != null && !overview.getAinfo().isBlank()) {
                    System.out.println("  Info: " + overview.getAinfo());
                }
                if (overview.getUrls() != null && !overview.getUrls().isEmpty()) {
                    for (OverviewUrl u : overview.getUrls()) {
                        if (u.getUrl() != null && !u.getUrl().isBlank()) {
                            String label = (u.getLabel() != null && !u.getLabel().isBlank()) ? u.getLabel() : "URL";
                            System.out.println("  " + label + ": " + u.getUrl());
                        }
                    }
                }
                if (overview.getTags() != null && !overview.getTags().isEmpty()) {
                    System.out.println("  Tags: " + String.join(", ", overview.getTags()));
                }
                if (showRawDecrypted && overview.getPs() != null) {
                    System.out.println("  Password Strength: " + overview.getPs());
                }
                for (Map.Entry<String, Object> entry : overview.getAdditionalProperties().entrySet()) {
                    if (!showRawDecrypted && isInternalOverviewProperty(entry.getKey())) {
                        continue;
                    }
                    if (entry.getValue() != null && !String.valueOf(entry.getValue()).isBlank()) {
                        System.out.println("  Overview [" + entry.getKey() + "]: " + formatValue(entry.getKey(), entry.getValue()));
                    }
                }
                if (item.getCreated() != null || item.getUpdated() != null) {
                    List<String> timestamps = new ArrayList<>();
                    if (item.getCreated() != null) {
                        timestamps.add("Created: " + formatEpochSeconds(item.getCreated()));
                    }
                    if (item.getUpdated() != null) {
                        timestamps.add("Updated: " + formatEpochSeconds(item.getUpdated()));
                    }
                    System.out.println("  " + String.join(", ", timestamps));
                }

                // 4. Display Decrypted Details
                if (item.getD() != null && item.getK() != null) {
                    if (detailsError != null) {
                        System.out.println("  Failed to decrypt details: " + detailsError.getMessage());
                    } else if (details != null) {
                        if (showRawDecrypted) {
                            System.out.println("Decrypted: " + decryptedDetailsJson);
                        }

                        // A. Standard Web Form Fields
                        if (details.getFields() != null) {
                            for (ItemField field : details.getFields()) {
                                if (!showRawDecrypted && (field.getValue() == null || field.getValue().isBlank())) {
                                    countFields++;
                                    continue;
                                }
                                String idText = (showRawDecrypted && field.getId() != null) ? "[Id: " + field.getId() + "] " : "";
                                String fieldLabel = (field.getName() != null && !field.getName().isBlank())
                                        ? field.getName()
                                        : (field.getDesignation() != null && !field.getDesignation().isBlank() ? field.getDesignation() : "field");
                                System.out.print("  " + idText + fieldLabel + " = " + field.getValue());

                                if (showRawDecrypted) {
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
                                        System.out.print(")");
                                    }
                                }
                                System.out.println();
                                countFields++;
                            }
                        }

                        // B. Structured Sections & Section Fields
                        if (details.getSections() != null) {
                            for (ItemSection section : details.getSections()) {
                                if (section.getFields() == null || section.getFields().isEmpty()) {
                                    continue;
                                }
                                boolean hasExplicitTitle = section.getTitle() != null && !section.getTitle().isBlank();
                                String indent = "  ";
                                if (hasExplicitTitle) {
                                    System.out.println("  Section [" + section.getTitle() + "]:");
                                    indent = "    ";
                                } else if (showRawDecrypted) {
                                    String secName = (section.getName() != null && !section.getName().isBlank()) ? section.getName() : "Section";
                                    System.out.println("  Section [" + secName + "]:");
                                    indent = "    ";
                                }

                                for (ItemSectionField sField : section.getFields()) {
                                    String valStr = sField.getFormattedValue();
                                    if (!showRawDecrypted && (valStr == null || valStr.isBlank())) {
                                        countFields++;
                                        continue;
                                    }
                                    String label = (sField.getTitle() != null && !sField.getTitle().isBlank())
                                            ? sField.getTitle()
                                            : (sField.getName() != null && !sField.getName().isBlank() ? sField.getName() : "field");

                                    String metaSuffix = "";
                                    if (showRawDecrypted) {
                                        StringBuilder meta = new StringBuilder();
                                        if (sField.getKind() != null && !sField.getKind().isBlank()) {
                                            meta.append("Kind: ").append(sField.getKind());
                                        }
                                        if (sField.getName() != null && !sField.getName().isBlank() && !sField.getName().equals(label)) {
                                            if (!meta.isEmpty()) meta.append(", ");
                                            meta.append("Name: ").append(sField.getName());
                                        }
                                        if (!meta.isEmpty()) {
                                            metaSuffix = " (" + meta + ")";
                                        }
                                    }
                                    System.out.println(indent + label + " = " + valStr + metaSuffix);
                                    countFields++;
                                }
                            }
                        }

                        // C. Top-level Explicit Detail Properties
                        if (details.getPassword() != null && !details.getPassword().isEmpty()) {
                            System.out.println("  Password: " + details.getPassword());
                        }
                        if (details.getNumber() != null && !details.getNumber().isEmpty()) {
                            System.out.println("  Number: " + details.getNumber());
                        }
                        if (details.getMembership_no() != null && !details.getMembership_no().isEmpty()) {
                            System.out.println("  Membership No.: " + details.getMembership_no());
                        }
                        if (details.getNotesPlain() != null && !details.getNotesPlain().isEmpty()) {
                            System.out.println("  Notes: " + details.getNotesPlain());
                        }

                        // D. Catch-all Top-level Detail Properties (Passports, Credit Cards, Licenses, Routers, etc.)
                        for (Map.Entry<String, Object> entry : details.getAdditionalProperties().entrySet()) {
                            if (!showRawDecrypted && isInternalDetailProperty(entry.getKey())) {
                                continue;
                            }
                            Object val = entry.getValue();
                            if (val != null && !String.valueOf(val).isBlank()) {
                                System.out.println("  " + entry.getKey() + ": " + formatValue(entry.getKey(), val));
                            }
                        }

                        // E. Password History
                        if (details.getPasswordHistory() != null && !details.getPasswordHistory().isEmpty()) {
                            System.out.println("  Password History:");
                            for (PasswordHistoryEntry hist : details.getPasswordHistory()) {
                                String timeStr = hist.getTime() != null ? " (at " + formatEpochSeconds(hist.getTime()) + ")" : "";
                                System.out.println("    - " + hist.getValue() + timeStr);
                            }
                        }
                    }
                }
                System.out.println();
                System.out.println("------------------------------------------------------------------");
                System.out.println();

            } catch (Exception e) {
                System.out.println("Failed to read item " + item.getUuid() + ": " + e.getMessage());
            }
        }
    }

    private static boolean isInternalOverviewProperty(String key) {
        if (key == null) return false;
        return switch (key) {
            case "pbe", "pgrng", "ps" -> true;
            default -> false;
        };
    }

    private static boolean isInternalDetailProperty(String key) {
        if (key == null) return false;
        return switch (key) {
            case "backupKeys", "htmlForm", "htmlMethod", "htmlAction", "htmlId", "htmlName" -> true;
            default -> false;
        };
    }

    private static boolean matchesSearchTerms(
            VaultItem item,
            ItemOverview overview,
            String folderPath,
            ItemDetails details,
            List<String> searchTerms) {
        StringBuilder searchable = new StringBuilder();
        appendSearchable(searchable, overview.getTitle());
        appendSearchable(searchable, item.getCategory());
        appendSearchable(searchable, folderPath);
        appendSearchable(searchable, overview.getUrl());
        appendSearchable(searchable, overview.getAinfo());
        if (overview.getTags() != null) {
            overview.getTags().forEach(t -> appendSearchable(searchable, t));
        }
        if (overview.getUrls() != null) {
            for (OverviewUrl u : overview.getUrls()) {
                appendSearchable(searchable, u.getLabel());
                appendSearchable(searchable, u.getUrl());
            }
        }
        overview.getAdditionalProperties().values().forEach(v -> appendSearchable(searchable, v));

        if (details != null) {
            if (details.getFields() != null) {
                for (ItemField f : details.getFields()) {
                    appendSearchable(searchable, f.getName());
                    appendSearchable(searchable, f.getValue());
                }
            }
            if (details.getSections() != null) {
                for (ItemSection s : details.getSections()) {
                    appendSearchable(searchable, s.getTitle());
                    appendSearchable(searchable, s.getName());
                    if (s.getFields() != null) {
                        for (ItemSectionField sf : s.getFields()) {
                            appendSearchable(searchable, sf.getTitle());
                            appendSearchable(searchable, sf.getName());
                            appendSearchable(searchable, sf.getFormattedValue());
                        }
                    }
                }
            }
            appendSearchable(searchable, details.getPassword());
            appendSearchable(searchable, details.getNumber());
            appendSearchable(searchable, details.getMembership_no());
            appendSearchable(searchable, details.getNotesPlain());
            details.getAdditionalProperties().forEach((k, v) -> {
                appendSearchable(searchable, k);
                appendSearchable(searchable, v);
            });
            if (details.getPasswordHistory() != null) {
                for (PasswordHistoryEntry h : details.getPasswordHistory()) {
                    appendSearchable(searchable, h.getValue());
                }
            }
        }

        String haystack = searchable.toString().toLowerCase();
        for (String term : searchTerms) {
            if (haystack.contains(term.toLowerCase())) {
                return true;
            }
        }
        return false;
    }

    private static void appendSearchable(StringBuilder sb, Object value) {
        if (value != null) {
            sb.append(value).append(' ');
        }
    }

    private static String formatEpochSeconds(long epochSeconds) {
        return TIMESTAMP_FORMATTER.format(Instant.ofEpochSecond(epochSeconds));
    }

    private static String formatValue(String key, Object value) {
        if (value instanceof Number num) {
            String lowerKey = key != null ? key.toLowerCase() : "";
            long l = num.longValue();
            if ((lowerKey.contains("date") || lowerKey.contains("time") || lowerKey.equals("expires") || lowerKey.equals("validfrom"))
                    && l > 100_000_000L && l < 10_000_000_000L) {
                return formatEpochSeconds(l) + " (" + l + ")";
            }
        }
        return String.valueOf(value);
    }
}
