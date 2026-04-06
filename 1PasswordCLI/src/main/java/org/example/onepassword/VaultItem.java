package org.example.onepassword;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.HashMap;

@JsonIgnoreProperties(ignoreUnknown = true)
public class VaultItem {
    private String uuid;
    private String hmac;
    private String o; // encrypted overview
    private String d; // encrypted details
    private String k; // encrypted item key
    private String category;

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

    public void setCategory(String category) {
        this.category = category;
    }
}
