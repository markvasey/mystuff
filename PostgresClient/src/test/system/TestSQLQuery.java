package system;

import database.SQLQuery;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;
import utils.AppProperties;

import java.util.List;
import java.util.Map;

public class TestSQLQuery {
    @Test
    public void SLQuery_RunSQL() {
        boolean returnValue = SQLQuery.RunSQL("SELECT * FROM public.teampeople");
        Assertions.assertTrue(returnValue);
    }

    @Test
    public void SLQuery_GetSQLCount() {
        int returnValue = SQLQuery.GetSQLCount("SELECT count(*) FROM public.teampeople");
        Assertions.assertEquals(15,returnValue);

        returnValue = SQLQuery.GetSQLCount("SELECT name FROM public.teampeople limit 1");
        Assertions.assertEquals(-1,returnValue);

        returnValue = SQLQuery.GetSQLCount("SELECT count(*), sum(grade) FROM public.teampeople");
        Assertions.assertEquals(-1,returnValue);
    }

    @Test
    public void SLQuery_GetSQLRows() {
        List<Map<String,Object>> rows = SQLQuery.GetSQLRows("SELECT * FROM public.teampeople");
        Assertions.assertEquals(15, rows.size());

        Map<String,Object> row = rows.getFirst();
        Assertions.assertTrue(row.containsKey("personid"));
        Assertions.assertTrue(row.containsKey("name"));
        Assertions.assertTrue(row.containsKey("managerid"));
        Assertions.assertTrue(row.containsKey("manager"));
        Assertions.assertTrue(row.containsKey("role"));
        Assertions.assertTrue(row.containsKey("grade"));

        Assertions.assertEquals(1,row.get("personid"));
        Assertions.assertEquals("Joe Smith",row.get("name"));
        Assertions.assertNull(row.get("managerid"));
        Assertions.assertNull(row.get("manager"));
        Assertions.assertEquals("Developer",row.get("role"));
        Assertions.assertEquals(13,row.get("grade"));

        row = rows.get(1);
        Assertions.assertEquals(2,row.get("personid"));
        Assertions.assertEquals("Jacqueline Ross",row.get("name"));
        Assertions.assertEquals(1,row.get("managerid"));
        Assertions.assertEquals("Joe Smith",row.get("manager"));
        Assertions.assertEquals("Developer",row.get("role"));
        Assertions.assertEquals(12,row.get("grade"));
    }
}
