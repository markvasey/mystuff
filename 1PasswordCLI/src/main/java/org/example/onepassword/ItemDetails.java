package org.example.onepassword;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
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
}
