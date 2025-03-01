import org.junit.Test;
import org.junit.jupiter.api.Assertions;
import utils.AppProperties;
import utils.Credential;
import utils.CredentialHelper;

public class TestAppProperties {
    @Test
    public void TestAppPropertiesetProperty() {
        String propertyValue = AppProperties.GetProperty("DBServerName");
        Assertions.assertEquals("localhost", propertyValue);

        propertyValue = AppProperties.GetProperty("DBServerPort");
        Assertions.assertEquals("5432", propertyValue);

        propertyValue = AppProperties.GetProperty("DatabaseName");
        Assertions.assertEquals("postgres", propertyValue);
    }
}
