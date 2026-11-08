package org.example.onepassword.dataClasses;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ItemDetails {
    @JsonProperty("fields")
    private List<ItemField> fields;

    @JsonProperty("sections")
    private List<ItemSection> sections;

    @JsonProperty("notesPlain")
    private String notesPlain;

    @JsonProperty("number")
    private String number;

    @JsonProperty("password")
    private String password;

    @JsonProperty("membership_no")
    private String membership_no;

    @JsonProperty("passwordHistory")
    private List<PasswordHistoryEntry> passwordHistory;

    private final Map<String, Object> additionalProperties = new LinkedHashMap<>();

    public List<ItemField> getFields() {
        return fields;
    }

    public void setFields(List<ItemField> fields) {
        this.fields = fields;
    }

    public List<ItemSection> getSections() {
        return sections;
    }

    public void setSections(List<ItemSection> sections) {
        this.sections = sections;
    }

    public String getNotesPlain() {
        return notesPlain;
    }

    public void setNotesPlain(String notesPlain) {
        this.notesPlain = notesPlain;
    }

    public String getNumber() {
        return number;
    }

    public void setNumber(String number) {
        this.number = number;
    }

    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    public String getMembership_no() {
        return membership_no;
    }

    public void setMembership_no(String membership_no) {
        this.membership_no = membership_no;
    }

    public List<PasswordHistoryEntry> getPasswordHistory() {
        return passwordHistory;
    }

    public void setPasswordHistory(List<PasswordHistoryEntry> passwordHistory) {
        this.passwordHistory = passwordHistory;
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
