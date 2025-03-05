import org.example.database.Credentials;
import org.example.database.CredentialsHelper;
import org.junit.Assert;
import org.junit.Test;

public class TestCredentialsHelper {
    @Test
    public void TestCredentialsHelper_LoadCredentials() {
        Credentials credentials = CredentialsHelper.LoadCredentials("creds.txt");
        Assert.assertNotNull(credentials);
        Assert.assertFalse(credentials.getUsername().isEmpty());
        Assert.assertFalse(credentials.getPassword().isEmpty());
    }
}
