package org.example.onepassword.processors;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.example.onepassword.dataClasses.VaultItem;
import org.example.onepassword.utils.Utils;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class VaultBandsProcessor {

    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static void ProcessVault(Path vaultPath, byte[] vaultMasterKey, byte[] vaultOverviewKey, String searchString) {
        List<String> searchTerms = Utils.parseSearchString(searchString);
        File defaultDir = vaultPath.resolve("default").toFile();
        File[] bandFiles = defaultDir.listFiles((_, name) -> name.startsWith("band_") && name.endsWith(".js"));

        if (bandFiles != null) {
            for (File bandFile : bandFiles) {
                if (bandFile.getName().contains("conflicted copy")) continue;

                try {
                    String bandJs = Files.readString(bandFile.toPath(), StandardCharsets.UTF_8);
                    int start = bandJs.indexOf("{");
                    int end = bandJs.lastIndexOf("}");
                    if (start == -1 || end == -1) continue;

                    String bandJson = bandJs.substring(start, end + 1);
                    Map<String, VaultItem> items = objectMapper.readValue(bandJson, new TypeReference<>() { });

                    DisplayProcessor.DisplayResults(items, vaultMasterKey, vaultOverviewKey, searchTerms);
                } catch (Exception e) {
                    System.out.println("Error reading band file: " + bandFile.getName() + ", " + e.getMessage());
                }
            }
        }
        DisplayProcessor.PrintTotals();
    }
}
