package com.markvasey.mysearch.model;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

@Entity
@Table(name = "scan_metadata")
public class ScanMetadata {

    @Id
    @Column(name = "source", nullable = false)
    private String source;

    @Column(name = "sync_token")
    private String syncToken;

    public ScanMetadata() {}

    public ScanMetadata(String source, String syncToken) {
        this.source = source;
        this.syncToken = syncToken;
    }

    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }

    public String getSyncToken() { return syncToken; }
    public void setSyncToken(String syncToken) { this.syncToken = syncToken; }
}
