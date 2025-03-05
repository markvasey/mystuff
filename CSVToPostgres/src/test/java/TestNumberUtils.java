import org.example.utils.NumberUtils;
import org.junit.Assert;
import org.junit.Test;

import java.text.SimpleDateFormat;
import java.util.Date;
public class TestNumberUtils {
    @Test
    public void TestNumberUtils_isInteger() {
        Assert.assertFalse(NumberUtils.isInteger(""));
        Assert.assertFalse(NumberUtils.isInteger("foo"));

        Assert.assertTrue(NumberUtils.isInteger("1"));
        Assert.assertTrue(NumberUtils.isInteger("1.0"));
        Assert.assertTrue(NumberUtils.isInteger("1.00"));
        Assert.assertTrue(NumberUtils.isInteger("123,456"));
        Assert.assertTrue(NumberUtils.isInteger("123,456.00"));

        Assert.assertFalse(NumberUtils.isInteger("0.1"));
        Assert.assertFalse(NumberUtils.isInteger("1.1"));
        Assert.assertFalse(NumberUtils.isInteger("123.1"));
        Assert.assertFalse(NumberUtils.isInteger("123,456.1"));
        Assert.assertFalse(NumberUtils.isInteger("123,456.12"));
    }


    @Test
    public void TestNumberUtils_getInteger() {
        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger(""));
        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger("foo"));

        Assert.assertEquals(1,NumberUtils.getInteger("1"));
        Assert.assertEquals(1,NumberUtils.getInteger("1.0"));
        Assert.assertEquals(1,NumberUtils.getInteger("1.00"));
        Assert.assertEquals(123456,NumberUtils.getInteger("123,456"));
        Assert.assertEquals(123456,NumberUtils.getInteger("123,456.00"));

        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger("0.1"));
        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger("1.1"));
        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger("123.1"));
        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger("123,456.1"));
        Assert.assertEquals(Integer.MIN_VALUE,NumberUtils.getInteger("123,456.12"));
    }

    @Test
    public void TestNumberUtils_isBoolean() {
        Assert.assertFalse(NumberUtils.isBoolean(""));
        Assert.assertFalse(NumberUtils.isBoolean("foo"));

        Assert.assertTrue(NumberUtils.isBoolean("Yes"));
        Assert.assertTrue(NumberUtils.isBoolean("yes"));
        Assert.assertTrue(NumberUtils.isBoolean("No"));
        Assert.assertTrue(NumberUtils.isBoolean("no"));
        Assert.assertTrue(NumberUtils.isBoolean("True"));
        Assert.assertTrue(NumberUtils.isBoolean("true"));
        Assert.assertTrue(NumberUtils.isBoolean("False"));
        Assert.assertTrue(NumberUtils.isBoolean("false"));
        Assert.assertTrue(NumberUtils.isBoolean("0"));
        Assert.assertTrue(NumberUtils.isBoolean("1"));

        Assert.assertFalse(NumberUtils.isBoolean("0.1"));
        Assert.assertFalse(NumberUtils.isBoolean("1.1"));
        Assert.assertFalse(NumberUtils.isBoolean("123.1"));
        Assert.assertFalse(NumberUtils.isBoolean("123,456.1"));
        Assert.assertFalse(NumberUtils.isBoolean("123,456.12"));
    }

    @Test
    public void TestNumberUtils_getBoolean() {
        Assert.assertFalse(NumberUtils.getBoolean(""));
        Assert.assertFalse(NumberUtils.getBoolean("foo"));

        Assert.assertTrue(NumberUtils.getBoolean("Yes"));
        Assert.assertTrue(NumberUtils.getBoolean("yes"));
        Assert.assertFalse(NumberUtils.getBoolean("No"));
        Assert.assertFalse(NumberUtils.getBoolean("no"));
        Assert.assertTrue(NumberUtils.getBoolean("True"));
        Assert.assertTrue(NumberUtils.getBoolean("true"));
        Assert.assertFalse(NumberUtils.getBoolean("False"));
        Assert.assertFalse(NumberUtils.getBoolean("false"));
        Assert.assertFalse(NumberUtils.getBoolean("0"));
        Assert.assertTrue(NumberUtils.getBoolean("1"));

        Assert.assertFalse(NumberUtils.getBoolean("0.1"));
        Assert.assertFalse(NumberUtils.getBoolean("1.1"));
        Assert.assertFalse(NumberUtils.getBoolean("123.1"));
        Assert.assertFalse(NumberUtils.getBoolean("123,456.1"));
        Assert.assertFalse(NumberUtils.getBoolean("123,456.12"));
    }


    @Test
    public void TestNumberUtils_isDate() {
        Assert.assertFalse(NumberUtils.isDate(""));
        Assert.assertFalse(NumberUtils.isDate("foo"));

        Assert.assertTrue(NumberUtils.isDate("24/01/2025"));

        Assert.assertFalse(NumberUtils.isDate("0.1"));
        Assert.assertFalse(NumberUtils.isDate("1.1"));
        Assert.assertFalse(NumberUtils.isDate("123.1"));
        Assert.assertFalse(NumberUtils.isDate("123,456.1"));
        Assert.assertFalse(NumberUtils.isDate("123,456.12"));
    }
    @Test
    public void TestNumberUtils_getDate() {
        SimpleDateFormat DateFormatter = new SimpleDateFormat("yyyy-MM-dd");

        Date date = NumberUtils.getDate("24/01/2025");
        Assert.assertEquals("2025-01-24",DateFormatter.format(date));

        date = NumberUtils.getDate("24/02/2025");
        Assert.assertEquals("2025-02-24",DateFormatter.format(date));

        date = NumberUtils.getDate("12/10/1980");
        Assert.assertEquals("1980-10-12",DateFormatter.format(date));

        SimpleDateFormat DateFormatter2 = new SimpleDateFormat("dd MMM yyyy");

        date = NumberUtils.getDate("24/01/2025");
        Assert.assertEquals("24 Jan 2025",DateFormatter2.format(date));

        date = NumberUtils.getDate("24/02/2025");
        Assert.assertEquals("24 Feb 2025",DateFormatter2.format(date));

        date = NumberUtils.getDate("12/10/1980");
        Assert.assertEquals("12 Oct 1980",DateFormatter2.format(date));
    }

    @Test
    public void TestNumberUtils_removeTrailingZeroes () {
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroes("123"));
        Assert.assertEquals("0123",NumberUtils.removeTrailingZeroes("0123"));
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroes("1230"));
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroes("12300"));
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroes("123000"));
    }

    @Test
    public void TestNumberUtils_removeTrailingZeroDecimalPlaces () {
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroDecimalPlaces("123"));
        Assert.assertEquals("0123",NumberUtils.removeTrailingZeroDecimalPlaces("0123"));
        Assert.assertEquals("1230",NumberUtils.removeTrailingZeroDecimalPlaces("1230"));
        Assert.assertEquals("12300",NumberUtils.removeTrailingZeroDecimalPlaces("12300"));
        Assert.assertEquals("123000",NumberUtils.removeTrailingZeroDecimalPlaces("123000"));

        Assert.assertEquals("123.1",NumberUtils.removeTrailingZeroDecimalPlaces("123.1"));
        Assert.assertEquals("123.1",NumberUtils.removeTrailingZeroDecimalPlaces("123.10"));
        Assert.assertEquals("123.1",NumberUtils.removeTrailingZeroDecimalPlaces("123.100"));
        Assert.assertEquals("123.1001",NumberUtils.removeTrailingZeroDecimalPlaces("123.1001"));
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroDecimalPlaces("123.0"));
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroDecimalPlaces("123.00"));
        Assert.assertEquals("123",NumberUtils.removeTrailingZeroDecimalPlaces("123.000"));
    }
}
