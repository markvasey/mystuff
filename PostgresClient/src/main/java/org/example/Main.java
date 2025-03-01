package org.example;

import utils.Credential;
import utils.CredentialHelper;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Arrays;
import java.util.Properties;

public class Main {
    public static void main(String[] args) {

        System.out.println("Hello, World!");

        Credential credential = CredentialHelper.LoadCredential();

        final String url = "jdbc:postgresql://localhost:5432/postgres";
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