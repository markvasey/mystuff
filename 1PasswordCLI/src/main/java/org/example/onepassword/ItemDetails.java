package org.example.onepassword;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ItemDetails {
    private List<ItemField> fields;

    public List<ItemField> getFields() {
        return fields;
    }

    public void setFields(List<ItemField> fields) {
        this.fields = fields;
    }

    @JsonProperty("notesPlain")
    private String notesPlain;

    @JsonProperty("number")
    private String number;

    @JsonProperty("password")
    private String password;

    @JsonProperty("membership_no")
    private String membership_no;

    //@JsonProperty("passwordHistory")
    //private String passwordHistory;

    //@JsonProperty("backupKeys")
    //private String backupKeys;


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
}
