package org;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.*;

public class CsvToJsonConverter {

    public static void Process() throws Exception {
        // Input CSV file
        String csvFile = "input.csv";
        List<String[]> csvData = GetResourceFIle.ReadCSV(csvFile);

        // Create a map to hold the children for each parent
        Map<String, List<String>> hierarchy = new HashMap<>();

        // Parse the CSV data into a hierarchy map
        for (String[] row : csvData) {
            String parent = row[0];
            String child = row[1];

            // Add child to the parent's list
            hierarchy.computeIfAbsent(parent, k -> new ArrayList<>()).add(child);
        }

        // Build the tree structure starting from "Stream Lead"
        ObjectNode jsonTree = buildJsonTree("Stream Lead", hierarchy);

        // Convert the tree to JSON
        ObjectMapper mapper = new ObjectMapper();
        String jsonOutput = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(jsonTree);

        // Output the resulting JSON
        System.out.println(jsonOutput);
    }


    // Recursively build the JSON tree
    private static ObjectNode buildJsonTree(String parentName, Map<String, List<String>> hierarchy) {
        ObjectMapper mapper = new ObjectMapper();
        ObjectNode node = mapper.createObjectNode();

        // Set the "name" of the node
        node.put("name", parentName);

        // If the current node has children, build them
        List<String> children = hierarchy.get(parentName);
        if (children != null && !children.isEmpty()) {
            ArrayNode childrenArray = mapper.createArrayNode();

            for (String child : children) {
                ObjectNode childNode = buildJsonTree(child, hierarchy);
                childrenArray.add(childNode);
            }
            node.set("children", childrenArray);
        }

        return node;
    }
}

