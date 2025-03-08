package org.example.utils;

import org.apache.commons.csv.CSVFormat; import org.apache.commons.csv.CSVRecord;
import org.example.database.Credentials;
import org.example.database.CredentialsHelper;
import org.example.database.PostgresDatabaseConnection;

import java.io.FileReader;
import java.io.IOException;
import java.sql.*;
import java.text.SimpleDateFormat;
import java.util.*;


public class CsvToSqlServer {
    private static final SimpleDateFormat DateFormatter = new SimpleDateFormat("yyyy-MM-dd");
    public static void ImportCSVFileToNewTable(String csvFilePath, String tableName) throws SQLException, IOException {

        FileReader fileReader = new FileReader(csvFilePath);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(fileReader);

        // Convert records to list to analyze the first few rows (for type inference)
        List<CSVRecord> recordList = new ArrayList<>();
        for (CSVRecord record : records) {
            recordList.add(record);
        }

        if (recordList.isEmpty()) {
            System.out.println("CSV file is empty.");
            return;
        }

        // Get column names from the CSV header
        String[] columnNames = recordList.getFirst().toMap().keySet().toArray(new String[0]);

        // Infer column data types
        CSVColumnAnalyzer.AnalyzeCSV(columnNames, recordList);

        // Establish database connection
        String connectionString = PostgresDatabaseConnection.GetConnectionString();
        Credentials credentials = CredentialsHelper.LoadCredentials("creds.txt");
        final Properties props = new Properties();
        props.setProperty("user", credentials.getUsername());
        props.setProperty("password", credentials.getPassword());
        Connection connection = DriverManager.getConnection(connectionString, props);

        // Create table based on CSV header and inferred data types
        createTable(connection, columnNames, tableName);

        // Insert data from CSV into SQL Server table
        int recordCount = 0;
        for (CSVRecord record : recordList) {
            insertRow(connection, tableName, columnNames, record);
            recordCount++;
        }
        System.out.println("Successfully inserted " + recordCount + " rows into table " + tableName);

        System.out.println("Data successfully inserted into SQL Server table.");

        connection.close();
    }


    // Create SQL Server table based on inferred column data types
    private static void createTable(Connection connection, String[] columnNames, String tableName) throws SQLException {
        Statement stmt = connection.createStatement();
        stmt.executeUpdate("DROP TABLE IF EXISTS " + tableName);

        StringBuilder createTableQuery = new StringBuilder("CREATE TABLE " + tableName + " (");

        for (String column : columnNames) {
            String strippedColumnName = GetStrippedColumn(column);

            String dataType = CSVColumnAnalyzer.getColumnType(column);
            String sqlDataType;
            switch(dataType ) {
                case "Boolean":
                    sqlDataType = "bit";
                    break;
                case "Date":
                    sqlDataType = "datetime";
                    break;
                case "Integer":
                    sqlDataType = "int";
                    break;
                case "Double":
                    int dps = CSVColumnAnalyzer.getMaxDecimalPlaces(column);
                    sqlDataType = "int";
                    if(dps > 0) {
                        sqlDataType = "decimal(18," + dps + ")";
                    }
                    break;
                case "String":
                    sqlDataType = "varchar";
                    int maxLength = CSVColumnAnalyzer.getMaxStringLength(column);
                    if(maxLength==0) {
                        maxLength = 5;
                    }
                    sqlDataType = sqlDataType + "(" + maxLength + ")";
                    break;
                default:
                    sqlDataType = "nvarchar(250)";
                    break;
            }

            String sNotNull = "";
            if(!CSVColumnAnalyzer.getHasBlankValues(column)) {
                sNotNull = " not";
            }

            createTableQuery.append(strippedColumnName).append(" ").append(sqlDataType).append(sNotNull).append(" null,").append(System.lineSeparator());
        }

        // Remove last comma and space
        createTableQuery.setLength(createTableQuery.length() - 2);
        createTableQuery.append(")");

        System.out.println(createTableQuery);

        // Execute the create table statement
        stmt = connection.createStatement();
        stmt.executeUpdate(createTableQuery.toString());
        System.out.println("Table: " + tableName + " - created successfully.");
    }

    // Insert a row of data into SQL Server table
    private static void insertRow(Connection connection, String tableName, String[] columnNames, CSVRecord record) throws SQLException {
        StringBuilder insertQuery = new StringBuilder("INSERT INTO " + tableName + " (");
        StringBuilder valuesQuery = new StringBuilder("VALUES (");

        for (String column : columnNames) {
            String dataType = CSVColumnAnalyzer.getColumnType(column);
            String strippedColumnName = GetStrippedColumn(column);

            insertQuery.append(strippedColumnName).append(", ");

            String value = record.get(column);
            String sqlValue = "";
            if(value.isEmpty()) {
                sqlValue = "null";
            } else {
                switch (dataType) {
                    case "Boolean" -> sqlValue = NumberUtils.getBoolean(value) ? "1" : "0";
                    case "Integer" -> sqlValue = String.valueOf(NumberUtils.getInteger(value));
                    case "Double" -> sqlValue = value.replaceAll(",", "");
                    case "Date" -> sqlValue = "'" + DateFormatter.format(NumberUtils.getDate(value)) + "'";
                    case "String" -> {
                        value = value.replaceAll("\r", "").replaceAll("\n", "");
                        if (value.isEmpty()) {
                            sqlValue = "null";
                        } else {
                            sqlValue = "'" + value + "'";
                        }
                    }
                }
            }
            if(sqlValue.isEmpty()) {
                System.out.println("Unexpected empty string");
            }
            valuesQuery.append(sqlValue).append(", ");
        }

        // Remove last comma and space
        insertQuery.setLength(insertQuery.length() - 2);
        valuesQuery.setLength(valuesQuery.length() - 2);

        insertQuery.append(") ");
        valuesQuery.append(");");

        String finalQuery = insertQuery.append(valuesQuery).toString();

        //System.out.println(finalQuery);

        // Execute the insert statement
        Statement stmt = connection.createStatement();
        stmt.executeUpdate(finalQuery);
    }

    private static String GetStrippedColumn(String column) {
        String strippedColumnName = column.replaceAll(" ", "");
        strippedColumnName = strippedColumnName.replaceAll("-","");
        strippedColumnName = strippedColumnName.replaceAll("\\.","");
        strippedColumnName = strippedColumnName.replaceAll("\\+","");
        strippedColumnName = strippedColumnName.replaceAll("\\(","");
        strippedColumnName = strippedColumnName.replaceAll("\\)","");
        strippedColumnName = strippedColumnName.replaceAll("�","");
        strippedColumnName = strippedColumnName.replaceAll("/","");
        strippedColumnName = strippedColumnName.replaceAll("\\?","");
        strippedColumnName = strippedColumnName.replaceAll("\\+","");

        return strippedColumnName;
    }

}
