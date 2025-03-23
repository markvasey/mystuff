package database;

import records.Person;
import utils.DBConnectionHelper;

import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RecordQueries {
    public static List<Person> GetPeople() {
        List<Person> people = null;

        try (Connection connection = DBConnectionHelper.GetDBConnection()) {
            Statement statement = connection.createStatement();

            String sql = "SELECT * FROM public.teampeople";
            ResultSet resultSet = statement.executeQuery(sql);
            ResultSetMetaData metaData = resultSet.getMetaData();
            int columnCount = metaData.getColumnCount();
            if(columnCount != 6) {
                System.out.println("SQLQuery.GetPeople : Query returned " + columnCount + " columns. Expected 6.");
            } else {
                people = new ArrayList<>();
                while (resultSet.next()) {
                    people.add(new Person(
                            resultSet.getInt("personId"),
                            resultSet.getString("name"),
                            resultSet.getInt("managerId"),
                            resultSet.getString("manager"),
                            resultSet.getString("role"),
                            resultSet.getInt("grade")));
                }
            }
        } catch(SQLException e) {
            System.out.println("SQLQuery.GetSQLCount : Error connecting to database: " + e);
        }

        return people;
    }
}
