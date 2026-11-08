package org.example.onepassword.dataClasses;

import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ItemSectionField {
    private static final DateTimeFormatter DATE_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd").withZone(ZoneId.systemDefault());

    @JsonProperty("k")
    private String kind;

    @JsonProperty("n")
    private String name;

    @JsonProperty("t")
    private String title;

    @JsonProperty("v")
    private Object value;

    @JsonProperty("a")
    private Map<String, Object> attributes;

    private final Map<String, Object> additionalProperties = new LinkedHashMap<>();

    public String getKind() {
        return kind;
    }

    public void setKind(String kind) {
        this.kind = kind;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public Object getValue() {
        return value;
    }

    public void setValue(Object value) {
        this.value = value;
    }

    public Map<String, Object> getAttributes() {
        return attributes;
    }

    public void setAttributes(Map<String, Object> attributes) {
        this.attributes = attributes;
    }

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperty(String name, Object value) {
        additionalProperties.put(name, value);
    }

    /**
     * Returns a human-readable string representation of the section field's value,
     * formatting epoch dates, YYYYMM monthYear values, and nested address maps cleanly.
     */
    public String getFormattedValue() {
        if (value == null) {
            return "";
        }
        if ("date".equalsIgnoreCase(kind) && value instanceof Number num) {
            long epochSeconds = num.longValue();
            return DATE_FORMATTER.format(Instant.ofEpochSecond(epochSeconds)) + " (" + epochSeconds + ")";
        }
        if ("monthYear".equalsIgnoreCase(kind) && value instanceof Number num) {
            int ym = num.intValue();
            if (ym > 190000 && ym < 220000) {
                int year = ym / 100;
                int month = ym % 100;
                return String.format("%02d/%04d", month, year);
            }
        }
        if (value instanceof Map<?, ?> map) {
            return map.entrySet().stream()
                    .filter(e -> e.getValue() != null && !String.valueOf(e.getValue()).isBlank())
                    .map(e -> e.getKey() + ": " + e.getValue())
                    .collect(Collectors.joining(", ", "{", "}"));
        }
        return String.valueOf(value);
    }
}
