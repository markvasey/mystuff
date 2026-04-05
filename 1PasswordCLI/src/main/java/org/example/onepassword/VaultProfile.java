package org.example.onepassword;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class VaultProfile {
    private String salt;
    private int iterations;
    private String masterKey;
    private String overviewKey;

    public String getSalt() {
        return salt;
    }

    public void setSalt(String salt) {
        this.salt = salt;
    }

    public int getIterations() {
        return iterations;
    }

    public void setIterations(int iterations) {
        this.iterations = iterations;
    }

    public String getMasterKey() {
        return masterKey;
    }

    public void setMasterKey(String masterKey) {
        this.masterKey = masterKey;
    }

    public String getOverviewKey() {
        return overviewKey;
    }

    public void setOverviewKey(String overviewKey) {
        this.overviewKey = overviewKey;
    }
}
