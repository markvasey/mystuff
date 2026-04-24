package com.markvasey.mysearch.service;

import com.markvasey.mysearch.model.ScanMetadata;
import com.markvasey.mysearch.model.SearchItem;
import com.markvasey.mysearch.repository.ScanMetadataRepository;
import com.markvasey.mysearch.repository.SearchItemRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.time.Instant;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class LocalDropboxSyncServiceTest {

    @Mock
    private SearchItemRepository searchItemRepository;

    @Mock
    private ScanMetadataRepository scanMetadataRepository;

    private LocalDropboxSyncService service;

    @TempDir
    Path tempDir;

    @BeforeEach
    public void setUp() {
        service = new LocalDropboxSyncService(searchItemRepository, scanMetadataRepository);
        service.setDropboxRootPath(tempDir.toString());
    }

    @Test
    public void testSync_ProcessesAllowedFiles() throws Exception {
        // Create an allowed file
        Path docFile = tempDir.resolve("test.txt");
        Files.writeString(docFile, "Some content");

        // Mock metadata to allow processing (last scan far in the past)
        when(scanMetadataRepository.findById(anyString())).thenReturn(Optional.of(new ScanMetadata("LOCAL_DROPBOX_LAST_SCAN", "2000-01-01T00:00:00")));
        when(searchItemRepository.findByExternalKey(anyString())).thenReturn(Optional.empty());

        CompletableFuture<Void> future = service.sync();
        future.join();

        // Verify file was saved
        verify(searchItemRepository, atLeastOnce()).save(any(SearchItem.class));
        verify(scanMetadataRepository, atLeastOnce()).save(argThat(m -> m.getSource().equals("LOCAL_DROPBOX_LAST_SCAN")));
    }

    @Test
    public void testSync_IgnoresRestrictedFiles() throws Exception {
        // Create an ignored file extension
        Path pdfFile = tempDir.resolve("ignore.pdf");
        Files.writeString(pdfFile, "PDF Content");
        
        // Create a code file
        Path jsFile = tempDir.resolve("script.js");
        Files.writeString(jsFile, "console.log('hi');");

        // Create a hidden directory file
        Path dotDir = tempDir.resolve(".git");
        Files.createDirectories(dotDir);
        Path hiddenFile = dotDir.resolve("config.txt");
        Files.writeString(hiddenFile, "hidden");

        when(scanMetadataRepository.findById(anyString())).thenReturn(Optional.of(new ScanMetadata("LOCAL_DROPBOX_LAST_SCAN", "2000-01-01T00:00:00")));

        CompletableFuture<Void> future = service.sync();
        future.join();

        // Verify NO files were saved
        verify(searchItemRepository, never()).save(any(SearchItem.class));
    }

    @Test
    public void testSync_IncrementalSync() throws Exception {
        // Create an old file
        Path oldFile = tempDir.resolve("old.txt");
        Files.writeString(oldFile, "Old content");
        Files.setLastModifiedTime(oldFile, FileTime.from(Instant.parse("2020-01-01T00:00:00Z")));

        // Create a new file
        Path newFile = tempDir.resolve("new.txt");
        Files.writeString(newFile, "New content");
        Files.setLastModifiedTime(newFile, FileTime.from(Instant.parse("2026-01-01T00:00:00Z")));

        // Mock last scan to be BETWEEN the two files
        when(scanMetadataRepository.findById(anyString())).thenReturn(Optional.of(new ScanMetadata("LOCAL_DROPBOX_LAST_SCAN", "2024-01-01T00:00:00")));
        when(searchItemRepository.findByExternalKey(anyString())).thenReturn(Optional.empty());

        CompletableFuture<Void> future = service.sync();
        future.join();

        // Verify only ONE file (the new one) was saved
        ArgumentCaptor<SearchItem> captor = ArgumentCaptor.forClass(SearchItem.class);
        verify(searchItemRepository, times(1)).save(captor.capture());
        assertEquals("new.txt", captor.getValue().getTitle());
    }

    @Test
    public void testSaveFileAsSearchItem_NewItem() {
        Path path = tempDir.resolve("test.txt");
        String content = "Hello World";
        BasicFileAttributes attrs = mock(BasicFileAttributes.class);
        when(attrs.lastModifiedTime()).thenReturn(FileTime.from(Instant.now()));

        when(searchItemRepository.findByExternalKey(anyString())).thenReturn(Optional.empty());

        service.saveFileAsSearchItem(path, content, attrs);

        ArgumentCaptor<SearchItem> captor = ArgumentCaptor.forClass(SearchItem.class);
        verify(searchItemRepository).save(captor.capture());

        SearchItem saved = captor.getValue();
        assertEquals("test.txt", saved.getTitle());
        assertEquals("LOCAL_DROPBOX", saved.getSource());
        assertEquals(content, saved.getContent());
        assertEquals(path.toAbsolutePath().toString(), saved.getExternalKey());
    }

    @Test
    public void testSaveFileAsSearchItem_UpdateExisting() {
        Path path = tempDir.resolve("update.txt");
        String content = "Updated Content";
        BasicFileAttributes attrs = mock(BasicFileAttributes.class);
        when(attrs.lastModifiedTime()).thenReturn(FileTime.from(Instant.now()));

        SearchItem existing = new SearchItem();
        existing.setTitle("Old Title");
        when(searchItemRepository.findByExternalKey(anyString())).thenReturn(Optional.of(existing));

        service.saveFileAsSearchItem(path, content, attrs);

        verify(searchItemRepository).save(existing);
        assertEquals("update.txt", existing.getTitle());
        assertEquals(content, existing.getContent());
    }
}
