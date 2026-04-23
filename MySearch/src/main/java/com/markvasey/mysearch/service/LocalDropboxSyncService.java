package com.markvasey.mysearch.service;

import com.markvasey.mysearch.model.ScanMetadata;
import com.markvasey.mysearch.model.SearchItem;
import com.markvasey.mysearch.repository.ScanMetadataRepository;
import com.markvasey.mysearch.repository.SearchItemRepository;
import org.apache.tika.Tika;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.io.File;
import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

@Service
public class LocalDropboxSyncService {

    private final SearchItemRepository searchItemRepository;
    private final ScanMetadataRepository scanMetadataRepository;
    private final Tika tika = new Tika();

    private static final String SOURCE_NAME = "LOCAL_DROPBOX";
    private static final String METADATA_KEY = "LOCAL_DROPBOX_LAST_SCAN";

    // Allowed extensions based on user requirements: text files and documents
    private static final Set<String> ALLOWED_EXTENSIONS = new HashSet<>(Arrays.asList(
            "txt", "rtf", "doc", "docx"
    ));

    // Specifically ignored extensions as per user request (xls, pdf, and code files)
    private static final Set<String> IGNORED_EXTENSIONS = new HashSet<>(Arrays.asList(
            "xls", "xlsx", "pdf", "class", "js", "xml", "json", "html", "htm", "css", "py", "java", "sh"
    ));

    public LocalDropboxSyncService(SearchItemRepository searchItemRepository, ScanMetadataRepository scanMetadataRepository) {
        this.searchItemRepository = searchItemRepository;
        this.scanMetadataRepository = scanMetadataRepository;
    }

    @Async
    public CompletableFuture<Void> sync() throws IOException {
        String userHome = System.getProperty("user.home");
        Path dropboxPath = Paths.get(userHome, "Dropbox");

        if (!Files.exists(dropboxPath)) {
            System.out.println("Local Dropbox folder not found at: " + dropboxPath + ". Skipping local sync.");
            return CompletableFuture.completedFuture(null);
        }

        System.out.println("Starting Local Dropbox sync from: " + dropboxPath);

        ScanMetadata metadata = scanMetadataRepository.findById(METADATA_KEY)
                .orElse(new ScanMetadata(METADATA_KEY, "1970-01-01T00:00:00"));

        LocalDateTime lastScan = LocalDateTime.parse(metadata.getSyncToken());
        long lastScanMillis = lastScan.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        
        LocalDateTime currentScanStart = LocalDateTime.now();

        Files.walkFileTree(dropboxPath, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                if (attrs.isRegularFile()) {
                    String fileName = file.getFileName().toString().toLowerCase();
                    String extension = getExtension(fileName);

                    if (ALLOWED_EXTENSIONS.contains(extension) && !IGNORED_EXTENSIONS.contains(extension)) {
                        long lastModifiedMillis = attrs.lastModifiedTime().toMillis();
                        
                        // Check if file is new or modified since last scan
                        if (lastModifiedMillis > lastScanMillis) {
                            processFile(file, attrs);
                        }
                    }
                }
                return FileVisitResult.CONTINUE;
            }

            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                // Ignore hidden directories like .git
                if (dir.getFileName() != null && dir.getFileName().toString().startsWith(".")) {
                    return FileVisitResult.SKIP_SUBTREE;
                }
                return FileVisitResult.CONTINUE;
            }
        });

        updateMetadata(currentScanStart);
        System.out.println("Local Dropbox sync completed.");
        return CompletableFuture.completedFuture(null);
    }

    private void processFile(Path path, BasicFileAttributes attrs) {
        String externalKey = path.toAbsolutePath().toString();
        
        try {
            // If it exists, we could check if it's modified, but for simplicity we'll just skip if key exists
            // and lastModified is checked during walk. Actually, if we want to update, we'd delete first.
            // But usually personal docs don't change UUIDs.
            
            String content = extractContent(path);
            if (content == null || content.trim().isEmpty()) {
                return;
            }

            saveFileAsSearchItem(path, content, attrs);
            System.out.println("  Indexed local file: " + path.getFileName());
            
        } catch (Exception e) {
            System.err.println("  Failed to process local file " + path + ": " + e.getMessage());
        }
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void saveFileAsSearchItem(Path path, String content, BasicFileAttributes attrs) {
        String externalKey = path.toAbsolutePath().toString();
        
        // Find existing or create new
        SearchItem item = searchItemRepository.findByExternalKey(externalKey)
                .orElse(new SearchItem());

        item.setExternalKey(externalKey);
        item.setSource(SOURCE_NAME);
        item.setTitle(path.getFileName().toString());
        
        LocalDateTime fileDate = LocalDateTime.ofInstant(attrs.lastModifiedTime().toInstant(), ZoneId.systemDefault());
        item.setItemDate(fileDate);
        item.setScannedAt(LocalDateTime.now());
        
        item.setContent(content);
        item.setSnippet(createSnippet(content));

        searchItemRepository.save(item);
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void updateMetadata(LocalDateTime scanTime) {
        ScanMetadata metadata = scanMetadataRepository.findById(METADATA_KEY)
                .orElse(new ScanMetadata(METADATA_KEY, scanTime.toString()));
        metadata.setSyncToken(scanTime.toString());
        scanMetadataRepository.save(metadata);
    }

    private String extractContent(Path path) {
        try {
            return tika.parseToString(path);
        } catch (Exception e) {
            System.err.println("    Tika failed to parse " + path.getFileName() + ": " + e.getMessage());
            return null;
        }
    }

    private String getExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        return (dotIndex == -1) ? "" : fileName.substring(dotIndex + 1);
    }

    private String createSnippet(String content) {
        if (content == null) return "";
        String[] words = content.trim().split("\\s+");
        StringBuilder snippet = new StringBuilder();
        for (int i = 0; i < Math.min(words.length, 100); i++) {
            snippet.append(words[i]).append(" ");
        }
        return snippet.toString().trim();
    }
}
