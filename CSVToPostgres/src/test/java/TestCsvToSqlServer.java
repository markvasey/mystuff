import org.example.utils.CsvToSqlServer;
import org.example.utils.NumberUtils;
import org.junit.Test;
import org.junit.Assert;


import java.io.IOException;
import java.sql.SQLException;

import static org.junit.Assert.fail;

public class TestCsvToSqlServer {
    @Test
    public void TestCsvToSqlServer_ImportCSVFileToNewTable_HCM012StaffList() throws SQLException, IOException {
        CsvToSqlServer.ImportCSVFileToNewTable("/home/markvasey/Dropbox/GitHub/mystuff/Data/sample.csv", "Sample");
    }


    @Test
    public void TestCsvToSqlServer_insertRow () {
        String value = "97,000.00";
        String sqlValue = NumberUtils.removeTrailingZeroDecimalPlaces(value.replaceAll(",",""));
        Assert.assertEquals("97000",sqlValue);
    }
}
