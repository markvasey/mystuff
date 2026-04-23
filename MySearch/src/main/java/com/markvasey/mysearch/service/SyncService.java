package com.markvasey.mysearch.service;

import com.markvasey.mysearch.model.ScanMetadata;
import com.markvasey.mysearch.repository.ScanMetadataRepository;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.atomic.AtomicBoolean;

@Service
public class SyncService {

    private final YahooSyncService yahooSyncService;
    private final ScanMetadataRepository scanMetadataRepository;
    private final AtomicBoolean isSyncing = new AtomicBoolean(false);
    private LocalDateTime lastSyncTime;

    public SyncService(YahooSyncService yahooSyncService, ScanMetadataRepository scanMetadataRepository) {
        this.yahooSyncService = yahooSyncService;
        this.scanMetadataRepository = scanMetadataRepository;
        // Pre-load last sync time from DB if it exists
        this.scanMetadataRepository.findById("LAST_SUCCESSFUL_SYNC")
                .ifPresent(m -> this.lastSyncTime = LocalDateTime.parse(m.getSyncToken()));
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onStartup() {
        System.out.println("Application started. Triggering startup sync...");
        triggerSync();
    }

    @Async
    public void triggerSync() {
        if (isSyncing.get()) {
            System.out.println("Sync already in progress. Skipping.");
            return;
        }

        try {
            isSyncing.set(true);
            System.out.println("BACKGROUND SYNC: Starting...");
            yahooSyncService.sync();
            
            // Record successful sync time
            this.lastSyncTime = LocalDateTime.now();
            ScanMetadata syncTimeMetadata = new ScanMetadata("LAST_SUCCESSFUL_SYNC", lastSyncTime.toString());
            scanMetadataRepository.save(syncTimeMetadata);
            
            System.out.println("BACKGROUND SYNC: Completed at " + lastSyncTime);
        } catch (Exception e) {
            System.err.println("BACKGROUND SYNC: Error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            isSyncing.set(false);
            System.out.println("BACKGROUND SYNC: State reset to idle.");
        }
    }

    public boolean isSyncing() {
        return isSyncing.get();
    }

    public String getLastSyncTimeFormatted() {
        if (lastSyncTime == null) return "Never";
        return lastSyncTime.format(DateTimeFormatter.ofPattern("dd MMM yyyy HH:mm:ss"));
    }
}
