package org.example.onepassword.processors;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.dataClasses.VaultItem;
import org.example.onepassword.utils.Utils;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class VaultBandsProcessor {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static void ProcessVault(Path vaultPath, byte[] vaultMasterKey, byte[] vaultOverviewKey, String searchString) {
        ProcessVault(vaultPath, vaultMasterKey, vaultOverviewKey, searchString, false, true);
    }

    public static void ProcessVault(
            Path vaultPath,
            byte[] vaultMasterKey,
            byte[] vaultOverviewKey,
            String searchString,
            boolean showRawDecrypted,
            boolean filterTrash) {
        DisplayProcessor.resetCounters();
        List<String> searchTerms = Utils.parseSearchString(searchString);
        VaultFoldersProcessor.loadFolders(vaultPath, vaultOverviewKey);

        File defaultDir = vaultPath.resolve("default").toFile();
        File[] bandFiles = defaultDir.listFiles((_, name) -> name.startsWith("band_") && name.endsWith(".js"));

        if (bandFiles != null) {
            // Sort so standard band_0.js..band_F.js are processed before any "conflicted copy" files
            Arrays.sort(bandFiles, Comparator
                    .comparing((File f) -> f.getName().contains("conflicted copy"))
                    .thenComparing(File::getName));

            Map<String, VaultItem> mergedItems = new LinkedHashMap<>();

            for (File bandFile : bandFiles) {
                boolean isConflicted = bandFile.getName().contains("conflicted copy");

                try {
                    String bandJs = Files.readString(bandFile.toPath(), StandardCharsets.UTF_8);
                    int start = bandJs.indexOf("{");
                    int end = bandJs.lastIndexOf("}");
                    if (start == -1 || end == -1) continue;

                    String bandJson = bandJs.substring(start, end + 1);
                    Map<String, VaultItem> items = objectMapper.readValue(bandJson, new TypeReference<>() { });

                    for (Map.Entry<String, VaultItem> entry : items.entrySet()) {
                        String uuid = entry.getKey();
                        VaultItem item = entry.getValue();
                        if (item.getUuid() == null) {
                            item.setUuid(uuid);
                        }
                        VaultItem existing = mergedItems.get(uuid);
                        if (existing == null) {
                            mergedItems.put(uuid, item);
                        } else {
                            long existingUpdated = existing.getUpdated() != null ? existing.getUpdated() : 0L;
                            long candidateUpdated = item.getUpdated() != null ? item.getUpdated() : 0L;
                            if (candidateUpdated > existingUpdated) {
                                mergedItems.put(uuid, item);
                            }
                        }
                    }
                } catch (Exception e) {
                    if (!isConflicted) {
                        System.out.println("Error reading band file: " + bandFile.getName() + ", " + e.getMessage());
                    }
                }
            }

            if (!mergedItems.isEmpty()) {
                DisplayProcessor.DisplayResults(mergedItems, vaultMasterKey, vaultOverviewKey, searchTerms, showRawDecrypted, filterTrash);
            }
        }
        DisplayProcessor.PrintTotals();
    }
}
