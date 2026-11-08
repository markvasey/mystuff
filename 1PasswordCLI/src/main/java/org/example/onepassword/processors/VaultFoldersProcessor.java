package org.example.onepassword.processors;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.OpVaultDecryptor;
import org.example.onepassword.dataClasses.ItemOverview;
import org.example.onepassword.dataClasses.VaultFolder;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class VaultFoldersProcessor {

    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static Map<String, VaultFolder> folders = new HashMap<>();

    public static void loadFolders(Path vaultPath, byte[] vaultOverviewKey) {
        folders = new HashMap<>();
        if (vaultPath == null || vaultOverviewKey == null) {
            return;
        }
        Path foldersPath = vaultPath.resolve("default/folders.js");
        if (!Files.exists(foldersPath)) {
            return;
        }

        try {
            String foldersJs = Files.readString(foldersPath, StandardCharsets.UTF_8);
            int start = foldersJs.indexOf("{");
            int end = foldersJs.lastIndexOf("}");
            if (start == -1 || end == -1) {
                return;
            }
            String foldersJson = foldersJs.substring(start, end + 1);
            Map<String, VaultFolder> parsedFolders = objectMapper.readValue(foldersJson, new TypeReference<>() { });

            for (Map.Entry<String, VaultFolder> entry : parsedFolders.entrySet()) {
                VaultFolder folder = entry.getValue();
                if (folder.getUuid() == null) {
                    folder.setUuid(entry.getKey());
                }
                if (folder.getOverview() != null) {
                    try {
                        byte[] overviewData = Base64.getDecoder().decode(folder.getOverview());
                        byte[] decryptedOverview = OpVaultDecryptor.decryptOpData(overviewData, vaultOverviewKey);
                        ItemOverview overview = objectMapper.readValue(decryptedOverview, ItemOverview.class);
                        if (overview.getTitle() != null && !overview.getTitle().isBlank()) {
                            folder.setDecryptedTitle(overview.getTitle());
                        }
                    } catch (Exception ignored) {
                        // Keep folder without decryptedTitle if overview decryption fails
                    }
                }
                folders.put(entry.getKey(), folder);
            }
        } catch (Exception e) {
            System.out.println("Error reading folders.js: " + e.getMessage());
        }
    }

    public static Map<String, VaultFolder> getFolders() {
        return folders;
    }

    public static void clearFolders() {
        folders.clear();
    }

    /**
     * Resolves a folder UUID to its full hierarchical folder path (e.g., "Parent / Child").
     */
    public static String getFolderPath(String folderUuid) {
        if (folderUuid == null || folderUuid.isBlank()) {
            return null;
        }
        VaultFolder folder = folders.get(folderUuid);
        if (folder == null) {
            return folderUuid;
        }

        StringBuilder pathBuilder = new StringBuilder();
        Set<String> visited = new HashSet<>();
        VaultFolder current = folder;
        while (current != null && current.getUuid() != null && visited.add(current.getUuid())) {
            String name = current.getDecryptedTitle() != null ? current.getDecryptedTitle() : current.getUuid();
            if (pathBuilder.isEmpty()) {
                pathBuilder.insert(0, name);
            } else {
                pathBuilder.insert(0, name + " / ");
            }
            current = current.getParent() != null ? folders.get(current.getParent()) : null;
        }
        return pathBuilder.isEmpty() ? folderUuid : pathBuilder.toString();
    }
}
