import org.example.database.Credentials;
import org.example.utils.Files;
import org.junit.Assert;
import org.junit.Test;

public class TestFiles {
    @Test
    public void TestFiles_ReadResourcesFile() {
        String fileContents = Files.ReadResourcesFile("creds.txt");
        Assert.assertFalse(fileContents.isEmpty());
        Assert.assertTrue(fileContents.contains("="));
        Assert.assertTrue(fileContents.contains(";"));
    }
}

