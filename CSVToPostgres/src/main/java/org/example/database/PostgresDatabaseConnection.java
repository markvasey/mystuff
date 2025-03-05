package org.example.database;

import org.example.utils.AppProperties;

import java.sql.*;
import java.util.Properties;

public class PostgresDatabaseConnection {
    // Connect to your database.
    // Replace server name, username, and password with your credentials
    public static void TryConnect() {

        String connectionUrl = GetConnectionString();

        Credentials credentials = CredentialsHelper.LoadCredentials("creds.txt");
        final Properties props = new Properties();
        props.setProperty("user", credentials.getUsername());
        props.setProperty("password", credentials.getPassword());

        try (Connection connection = DriverManager.getConnection(connectionUrl, props);
             Statement statement = connection.createStatement()) {
            // Code here.
            ResultSet resultSet = statement.executeQuery("SELECT * FROM pg_catalog.pg_tables WHERE schemaname='public';");
            while (resultSet.next()) {
                System.out.println(resultSet.getString(1));
            }
        }
        // Handle any errors that may have occurred.
        catch (SQLException e) {
            e.printStackTrace();
        }
    }

    public static String GetConnectionString() {
        String serverName = AppProperties.GetProperty("DBServerName");
        String port = AppProperties.GetProperty("DBServerPort");
        String databaseName = AppProperties.GetProperty("DatabaseName");

        return "jdbc:postgresql://" + serverName + ":" + port + "/" + databaseName;

    }

}