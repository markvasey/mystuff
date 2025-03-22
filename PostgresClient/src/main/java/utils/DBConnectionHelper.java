package utils;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.util.Properties;

public class DBConnectionHelper {
    public static String GetDBConnectionURL(){
        String server = AppProperties.GetProperty("DBServerName");
        String port = AppProperties.GetProperty("DBServerPort");
        String database = AppProperties.GetProperty("DatabaseName");

        return "jdbc:postgresql://" + server + ":" + port + "/" + database;

    }

    public static Properties GetDBConnectionProperties() {
        Credential credential = CredentialHelper.LoadCredential();

        final Properties props = new Properties();
        props.setProperty("user", credential.Username);
        props.setProperty("password", credential.getPassword());

        return props;
    }

    public static Connection GetDBConnection() throws SQLException {
        final String url = DBConnectionHelper.GetDBConnectionURL();
        final Properties props = DBConnectionHelper.GetDBConnectionProperties();

        return DriverManager.getConnection(url, props);
    }
}
