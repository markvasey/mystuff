package org.example.onepassword.dataClasses;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.LinkedHashMap;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class VaultFolder {
    @JsonProperty("uuid")
    private String uuid;

    @JsonProperty("parent")
    private String parent;

    @JsonProperty("overview")
    private String overview;

    @JsonProperty("created")
    private Long created;

    @JsonProperty("updated")
    private Long updated;

    @JsonProperty("tx")
    private Long tx;

    @JsonProperty("trashed")
    private Boolean trashed;

    private String decryptedTitle;

    private final Map<String, Object> additionalProperties = new LinkedHashMap<>();

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public String getParent() {
        return parent;
    }

    public void setParent(String parent) {
        this.parent = parent;
    }

    public String getOverview() {
        return overview;
    }

    public void setOverview(String overview) {
        this.overview = overview;
    }

    public Long getCreated() {
        return created;
    }

    public void setCreated(Long created) {
        this.created = created;
    }

    public Long getUpdated() {
        return updated;
    }

    public void setUpdated(Long updated) {
        this.updated = updated;
    }

    public Long getTx() {
        return tx;
    }

    public void setTx(Long tx) {
        this.tx = tx;
    }

    public Boolean getTrashed() {
        return trashed;
    }

    public void setTrashed(Boolean trashed) {
        this.trashed = trashed;
    }

    public String getDecryptedTitle() {
        return decryptedTitle;
    }

    public void setDecryptedTitle(String decryptedTitle) {
        this.decryptedTitle = decryptedTitle;
    }

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }
}
