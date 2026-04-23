package com.markvasey.mysearch.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.UUID;

@Entity
@Table(name = "search_items")
public class SearchItem {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @Column(name = "external_key", nullable = false)
    private String externalKey;

    @Column(name = "source", nullable = false)
    private String source;

    @Column(name = "title")
    private String title;

    @Column(name = "content", columnDefinition = "TEXT")
    private String content;

    @Column(name = "snippet", columnDefinition = "TEXT")
    private String snippet;

    @Column(name = "item_date")
    private LocalDateTime itemDate;

    @Column(name = "scanned_at")
    private LocalDateTime scannedAt;

    // Generated search vector for Postgres FTS.
    @org.hibernate.annotations.GeneratedColumn(value = "setweight(to_tsvector('english', coalesce(title, '')), 'A') || setweight(to_tsvector('english', coalesce(content, '')), 'B')")
    @Column(name = "search_vector", columnDefinition = "tsvector")
    private String searchVector;

    public SearchItem() {}

    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    
    public String getExternalKey() { return externalKey; }
    public void setExternalKey(String externalKey) { this.externalKey = externalKey; }

    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getContent() { return content; }
    public void setContent(String content) { this.content = content; }

    public String getSnippet() { return snippet; }
    public void setSnippet(String snippet) { this.snippet = snippet; }

    public LocalDateTime getItemDate() { return itemDate; }
    public void setItemDate(LocalDateTime itemDate) { this.itemDate = itemDate; }

    public LocalDateTime getScannedAt() { return scannedAt; }
    public void setScannedAt(LocalDateTime scannedAt) { this.scannedAt = scannedAt; }
}
