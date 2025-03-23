package database;

import utils.DBConnectionHelper;

import java.sql.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SQLQuery {
    public static int GetSQLCount(String sql) {
        int returnValue = -1;
        try (Connection connection = DBConnectionHelper.GetDBConnection()) {
            Statement statement = connection.createStatement();
            ResultSet resultSet = statement.executeQuery(sql);
            int columns = resultSet.getMetaData().getColumnCount();
            int columnType = resultSet.getMetaData().getColumnType(1);
            if(columns>1 || columnType != -5) {
                System.out.println("SQLQuery.GetSQLCount : Query returned " + columns + " columns. Expected 1.");
            } else {
                while (resultSet.next()) {
                    returnValue = resultSet.getInt(1);
                }
            }
        } catch(SQLException e) {
            System.out.println("SQLQuery.GetSQLCount : Error connecting to database: " + e);
        }
        return returnValue;
    }

    public static boolean RunSQL(String sql) {
        boolean returnVale = false;
        try (Connection connection = DBConnectionHelper.GetDBConnection()) {
            Statement statement = connection.createStatement();
            returnVale = statement.execute(sql);
        } catch(SQLException e) {
            System.out.println("SQLQuery.RunSQL : Error connecting to database: " + e);
        }
        return returnVale;
    }

    public static List<Map<String,Object>> GetSQLRows(String sql) {
        List<Map<String,Object>> rows = null;

        try (Connection connection = DBConnectionHelper.GetDBConnection()) {
            Statement statement = connection.createStatement();

            ResultSet resultSet = statement.executeQuery(sql);
            ResultSetMetaData metaData = resultSet.getMetaData();
            int columnCount = metaData.getColumnCount();
            if(columnCount== 0) {
                System.out.println("SQLQuery.GetSQLCount : Query returned " + columnCount + " columns. Expected >0.");
            } else {
                rows = new ArrayList<>();
                while (resultSet.next()) {
                    Map<String, Object> row = new HashMap<>();
                    for(int i = 1; i<=columnCount; i++) {
                        row.put(metaData.getColumnName(i), resultSet.getObject(i));
                    }
                    rows.add(row);
                }
            }
        } catch(SQLException e) {
            System.out.println("SQLQuery.GetSQLCount : Error connecting to database: " + e);
        }

        return rows;
    }
}
