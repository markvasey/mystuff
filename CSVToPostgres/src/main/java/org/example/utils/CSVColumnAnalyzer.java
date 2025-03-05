package org.example.utils;

import org.apache.commons.csv.CSVRecord;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CSVColumnAnalyzer {
    private static final Map<String, String> ColumnTypes = new HashMap<>();
    private static final Map<String, Integer> MaxStringLength = new HashMap<>();
    private static final Map<String, Integer> MaxDecimalPlaces = new HashMap<>();
    private static final Map<String, Boolean> HasBlankValues = new HashMap<>();

    public static void AnalyzeCSV(String[] columnNames, List<CSVRecord> records) {

        ColumnTypes.clear();
        MaxStringLength.clear();
        MaxDecimalPlaces.clear();
        HasBlankValues.clear();

        // Initialize column stats
        for (String column : columnNames) {
            ColumnTypes.put(column, "");
            MaxStringLength.put(column, 0);
            MaxDecimalPlaces.put(column, 0);
            HasBlankValues.put(column, false);
        }

        // Analyze each row and column
        for (CSVRecord record : records) {
            for (String column : columnNames) {
                String cellValue = record.get(column);

                // Check for blank values
                if (cellValue.isEmpty()) {
                    HasBlankValues.put(column, true);
                } else {

                    // Attempt to determine the type
                    if (NumberUtils.isInteger(cellValue)) {
                        if (ColumnTypes.get(column).isEmpty()) {
                            ColumnTypes.put(column, "Integer");
                        }
                    } else if (NumberUtils.isDouble(cellValue)) {
                        int dps = 0;
                        int index = cellValue.indexOf(".");
                        if (index > -1) {
                            dps = cellValue.length() - 1 - index;
                        }

                        MaxDecimalPlaces.put(column, Math.max(MaxDecimalPlaces.get(column), dps));
                        if (ColumnTypes.get(column).isEmpty() || ColumnTypes.get(column).equals("Integer")) {
                            ColumnTypes.put(column, "Double");
                        }
                    } else if (NumberUtils.isBoolean(cellValue)) {
                        ColumnTypes.put(column, "Boolean");
                    } else if (NumberUtils.isDate(cellValue)) {
                        ColumnTypes.put(column, "Date");
                    } else {
                        if (!ColumnTypes.get(column).equals("String")) {
                            ColumnTypes.put(column, "String");
                        }
                        // Update max string length
                        MaxStringLength.put(column, Math.max(MaxStringLength.get(column), cellValue.length()));
                    }
                }
            }
        }
    }

    public static String getColumnType(String column) {
        return ColumnTypes.get(column);
    }

    public static int getMaxStringLength(String column) {
        return MaxStringLength.get(column);
    }
    public static int getMaxDecimalPlaces(String column) {
        return MaxDecimalPlaces.get(column);
    }
    public static boolean getHasBlankValues(String column) {
        return HasBlankValues.get(column);
    }
}

