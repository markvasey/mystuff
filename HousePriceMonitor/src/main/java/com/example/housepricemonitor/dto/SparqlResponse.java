package com.example.housepricemonitor.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Map;

public class SparqlResponse {
    private Head head;
    private Results results;

    public static class Head {
        private List<String> vars;
        public List<String> getVars() { return vars; }
        public void setVars(List<String> vars) { this.vars = vars; }
    }

    public static class Results {
        private List<Map<String, Binding>> bindings;
        public List<Map<String, Binding>> getBindings() { return bindings; }
        public void setBindings(List<Map<String, Binding>> bindings) { this.bindings = bindings; }
    }

    public static class Binding {
        private String type;
        private String value;
        private String datatype;

        public String getType() { return type; }
        public void setType(String type) { this.type = type; }
        public String getValue() { return value; }
        public void setValue(String value) { this.value = value; }
        public String getDatatype() { return datatype; }
        public void setDatatype(String datatype) { this.datatype = datatype; }
    }

    public Head getHead() { return head; }
    public void setHead(Head head) { this.head = head; }
    public Results getResults() { return results; }
    public void setResults(Results results) { this.results = results; }
}
