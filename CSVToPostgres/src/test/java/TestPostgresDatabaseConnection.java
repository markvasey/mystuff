import org.example.database.PostgresDatabaseConnection;
import org.junit.Test;

import static org.junit.Assert.*;

public class TestPostgresDatabaseConnection {
    @Test
    public void TestSQLDatabaseConnection_GetConnectionString() {
        String connectionString = PostgresDatabaseConnection.GetConnectionString();
        assertEquals("jdbc:postgresql://localhost:5432/postgres", connectionString);
    }
    @Test
    public void TestSQLDatabaseConnection_TryConnect() {
        try {
            PostgresDatabaseConnection.TryConnect();
        } catch (Exception e) {
            fail(e.getMessage());
        }
    }
}
