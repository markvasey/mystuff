import org.example.database.Credentials;
import org.junit.Assert;
import org.junit.Test;

public class TestCredentials {

    @Test
    public void TestCredentialsConstructor() {
        Credentials credentials = new Credentials("foo", "bar");
        Assert.assertNotNull(credentials);
    }

    @Test
    public void TestCredentialsGetters() {
        Credentials credentials = new Credentials("foo", "bar");
        Assert.assertNotNull(credentials);
        Assert.assertEquals("foo", credentials.getUsername());
        Assert.assertEquals("bar", credentials.getPassword());
    }
}
