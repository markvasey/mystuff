import org.junit.Assert;
import org.junit.Test;

public class StringTests {
    @Test
    public void testString1() {
        String str = "abc\\123";
        Assert.assertEquals(3, str.lastIndexOf("\\"));
    }

    @Test
    public void testString2() {
        String str = "abc//123";
        Assert.assertEquals(3, str.lastIndexOf("//"));
    }
}
