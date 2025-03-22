package org.example;

import utils.DBConnectionHelper;

import java.sql.Connection;
import java.sql.SQLException;

public class Main {
    public static void main(String[] args) {

        System.out.println("PostgresClient started");

        try (Connection connection = DBConnectionHelper.GetDBConnection()) {
            System.out.println(connection.getMetaData().getDatabaseProductVersion());
        } catch(SQLException e) {
            System.out.println("Error connecting to database: " + e);
        }

    }
}