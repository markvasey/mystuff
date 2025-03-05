import org.example.utils.AppProperties;
import org.junit.Assert;
import org.junit.Test;

public class TestAppProperties {
    @Test
    public void TestAppProperties_GetProperty() {
        String propertyValue = AppProperties.GetProperty("DBServerName");
        Assert.assertNotNull(propertyValue);
        Assert.assertEquals("localhost", propertyValue);
    }
}
