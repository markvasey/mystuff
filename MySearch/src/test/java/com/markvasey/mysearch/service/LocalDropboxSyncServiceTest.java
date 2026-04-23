package com.markvasey.mysearch.service;

import com.markvasey.mysearch.model.SearchItem;
import com.markvasey.mysearch.repository.ScanMetadataRepository;
import com.markvasey.mysearch.repository.SearchItemRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.time.Instant;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class LocalDropboxSyncServiceTest {

    @Mock
    private SearchItemRepository searchItemRepository;

    @Mock
    private ScanMetadataRepository scanMetadataRepository;

    private LocalDropboxSyncService service;

    @BeforeEach
    public void setUp() {
        service = new LocalDropboxSyncService(searchItemRepository, scanMetadataRepository);
    }

    @Test
    public void testSaveFileAsSearchItem_NewItem() {
        Path path = Paths.get("/home/user/Dropbox/test.txt");
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
        Path path = Paths.get("/home/user/Dropbox/update.txt");
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
