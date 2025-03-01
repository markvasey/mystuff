package org.example;

import utils.AppProperties;
import utils.Credential;
import utils.CredentialHelper;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

public class Main {
    public static void main(String[] args) {

        System.out.println("Hello, World!");

        Credential credential = CredentialHelper.LoadCredential();

        String server = AppProperties.GetProperty("DBServerName");
        String port = AppProperties.GetProperty("DBServerPort");
        String database = AppProperties.GetProperty("DatabaseName");

        final String url = "jdbc:postgresql://" + server + ":" + port + "/" + database;
        final Properties props = new Properties();
        props.setProperty("user", credential.Username);
        props.setProperty("password", credential.getPassword());
        try (Connection connection = DriverManager.getConnection(url, props)) {
            System.out.println(connection.getMetaData().getDatabaseProductVersion());
        } catch(SQLException e) {
            System.out.println("Error connecting to database: " + e);
        }

    }
}