package com.markvasey.mysearch.service;

import com.markvasey.mysearch.model.ScanMetadata;
import com.markvasey.mysearch.model.SearchItem;
import com.markvasey.mysearch.repository.ScanMetadataRepository;
import com.markvasey.mysearch.repository.SearchItemRepository;
import jakarta.mail.*;
import jakarta.mail.internet.MimeMultipart;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.Properties;
import java.util.concurrent.CompletableFuture;

@Service
public class YahooSyncService {

    private final SearchItemRepository searchItemRepository;
    private final ScanMetadataRepository scanMetadataRepository;

    @Value("${yahoo.mail.host}")
    private String host;

    @Value("${yahoo.mail.port}")
    private String port;

    @Value("${yahoo.mail.username}")
    private String username;

    @Value("${yahoo.mail.app-password}")
    private String appPassword;

    public YahooSyncService(SearchItemRepository searchItemRepository, ScanMetadataRepository scanMetadataRepository) {
        this.searchItemRepository = searchItemRepository;
        this.scanMetadataRepository = scanMetadataRepository;
    }

    @Async
    public CompletableFuture<Void> sync() throws Exception {
        if ("YOUR_YAHOO_EMAIL".equals(username)) {
            System.out.println("Yahoo credentials not configured. Skipping Yahoo sync.");
            return CompletableFuture.completedFuture(null);
        }

        Properties props = new Properties();
        props.put("mail.store.protocol", "imaps");
        props.put("mail.imaps.host", host);
        props.put("mail.imaps.port", port);
        props.put("mail.imaps.ssl.enable", "true");
        props.put("mail.imaps.timeout", "60000"); // 1 min timeout
        props.put("mail.imaps.connectiontimeout", "60000");

        Session session = Session.getInstance(props);
        Store store = session.getStore("imaps");
        store.connect(username, appPassword);

        try {
            Folder root = store.getDefaultFolder();
            syncFolderRecursive(root);
        } finally {
            store.close();
        }
        return CompletableFuture.completedFuture(null);
    }

    private void syncFolderRecursive(Folder folder) throws MessagingException {
        if ((folder.getType() & Folder.HOLDS_MESSAGES) != 0) {
            syncFolder(folder);
        }

        if ((folder.getType() & Folder.HOLDS_FOLDERS) != 0) {
            for (Folder subfolder : folder.list()) {
                syncFolderRecursive(subfolder);
            }
        }
    }

    private void syncFolder(Folder folder) {
        String folderFullName = folder.getFullName();
        try {
            folder.open(Folder.READ_ONLY);
            int totalMessages = folder.getMessageCount();
            
            if (totalMessages == 0) {
                folder.close(false);
                return;
            }

            System.out.println("Syncing folder: " + folderFullName + " (Total: " + totalMessages + ")");

            if (!(folder instanceof UIDFolder)) {
                System.err.println("Folder " + folderFullName + " does not support UIDs. Skipping.");
                folder.close(false);
                return;
            }
            UIDFolder uidFolder = (UIDFolder) folder;

            String metadataKey = "YAHOO_MAIL_" + folderFullName;
            ScanMetadata metadata = scanMetadataRepository.findById(metadataKey)
                    .orElse(new ScanMetadata(metadataKey, "0"));
            
            long lastUid = Long.parseLong(metadata.getSyncToken());
            Message[] messages;

            if (lastUid == 0) {
                // Fetch ALL messages from this folder to get full history
                messages = folder.getMessages();
                System.out.println("Full sync: Found " + messages.length + " messages in " + folderFullName);
            } else {
                // Incremental sync: Only new messages since last highest UID
                messages = uidFolder.getMessagesByUID(lastUid + 1, UIDFolder.MAXUID);
                System.out.println("Incremental sync: Found " + messages.length + " new messages in " + folderFullName);
            }

            if (messages.length > 0) {
                FetchProfile fp = new FetchProfile();
                fp.add(FetchProfile.Item.ENVELOPE);
                folder.fetch(messages, fp);

                long maxUid = lastUid;
                int savedCount = 0;
                int skipCount = 0;
                
                for (Message message : messages) {
                    try {
                        long currentUid = uidFolder.getUID(message);
                        if (currentUid > maxUid) {
                            maxUid = currentUid;
                        }
                        
                        String externalKey = folderFullName + "_" + currentUid;
                        if (searchItemRepository.existsByExternalKey(externalKey)) {
                            skipCount++;
                            continue;
                        }

                        saveMessageAsSearchItem(message, folderFullName, currentUid);
                        savedCount++;
                        
                        if (savedCount % 20 == 0) {
                            System.out.println("  " + folderFullName + ": Saved " + savedCount + ", Skipped " + skipCount + "/" + messages.length);
                        }
                    } catch (Exception e) {
                        System.err.println("  Error in " + folderFullName + " for a message: " + e.getMessage());
                    }
                }
                updateMetadata(metadataKey, maxUid);
                System.out.println("Finished " + folderFullName + ". Saved " + savedCount + " new items, Skipped " + skipCount + " existing.");
            }

            folder.close(false);
        } catch (Exception e) {
            System.err.println("Failed to sync folder " + folderFullName + ": " + e.getMessage());
        }
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void saveMessageAsSearchItem(Message message, String folderName, long uid) throws MessagingException, IOException {
        SearchItem item = new SearchItem();
        item.setExternalKey(folderName + "_" + uid);
        item.setSource("YAHOO_MAIL");
        item.setTitle("[" + folderName + "] " + message.getSubject());
        
        if (message.getSentDate() != null) {
            item.setItemDate(message.getSentDate().toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime());
        } else {
            item.setItemDate(LocalDateTime.now());
        }
        
        item.setScannedAt(LocalDateTime.now());
        
        String content = getTextFromMessage(message);
        item.setContent(content);
        item.setSnippet(createSnippet(content));

        searchItemRepository.save(item);
    }

    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void updateMetadata(String key, long maxUid) {
        ScanMetadata metadata = scanMetadataRepository.findById(key)
                .orElse(new ScanMetadata(key, "0"));
        metadata.setSyncToken(String.valueOf(maxUid));
        scanMetadataRepository.save(metadata);
    }

    private String getTextFromMessage(Message message) throws MessagingException, IOException {
        try {
            if (message.isMimeType("text/plain")) {
                Object content = message.getContent();
                return content != null ? content.toString() : "";
            } else if (message.isMimeType("multipart/*")) {
                MimeMultipart mimeMultipart = (MimeMultipart) message.getContent();
                return getTextFromMimeMultipart(mimeMultipart);
            }
        } catch (Exception e) {
            return "[Error extracting content: " + e.getMessage() + "]";
        }
        return "";
    }

    private String getTextFromMimeMultipart(MimeMultipart mimeMultipart) throws MessagingException, IOException {
        StringBuilder result = new StringBuilder();
        int count = mimeMultipart.getCount();
        for (int i = 0; i < count; i++) {
            BodyPart bodyPart = mimeMultipart.getBodyPart(i);
            if (bodyPart.isMimeType("text/plain")) {
                result.append(bodyPart.getContent());
            } else if (bodyPart.isMimeType("text/html")) {
                String html = (String) bodyPart.getContent();
                result.append(org.springframework.web.util.HtmlUtils.htmlUnescape(html.replaceAll("<[^>]*>", " ")));
            } else if (bodyPart.getContent() instanceof MimeMultipart) {
                result.append(getTextFromMimeMultipart((MimeMultipart) bodyPart.getContent()));
            }
        }
        return result.toString();
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
