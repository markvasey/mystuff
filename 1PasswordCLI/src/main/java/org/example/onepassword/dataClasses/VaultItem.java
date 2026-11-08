package org.example.onepassword.dataClasses;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import org.example.onepassword.utils.Utils;

import java.util.LinkedHashMap;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class VaultItem {
    private String uuid;
    private String hmac;
    private String o; // encrypted overview
    private String d; // encrypted details
    private String k; // encrypted item key
    private String category;
    private Long created;
    private Long updated;
    private Long tx;
    private Boolean trashed;
    private Integer fave;
    private String folder;

    private final Map<String, Object> additionalProperties = new LinkedHashMap<>();

    public String getUuid() {
        return uuid;
    }

    public void setUuid(String uuid) {
        this.uuid = uuid;
    }

    public String getO() {
        return o;
    }

    public void setO(String o) {
        this.o = o;
    }

    public String getD() {
        return d;
    }

    public void setD(String d) {
        this.d = d;
    }

    public String getK() {
        return k;
    }

    public void setK(String k) {
        this.k = k;
    }

    public String getHmac() {
        return hmac;
    }

    public void setHmac(String hmac) {
        this.hmac = hmac;
    }

    public String getCategory() {
        return Utils.getCategoryName(category);
    }

    public String getCategoryCode() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
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

    public Integer getFave() {
        return fave;
    }

    public void setFave(Integer fave) {
        this.fave = fave;
    }

    public String getFolder() {
        return folder;
    }

    public void setFolder(String folder) {
        this.folder = folder;
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
