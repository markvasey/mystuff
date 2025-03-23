package system;

import database.RecordQueries;
import database.SQLQuery;
import org.junit.Test;
import org.junit.jupiter.api.Assertions;
import records.Person;

import java.util.List;
import java.util.Map;

public class TestRecordQueries {


    @Test
    public void TestRecordQueriesGetPeople() {
        List<Person> people = RecordQueries.GetPeople();
        Assertions.assertEquals(15, people.size());

        Person person  = people.getFirst();
        Assertions.assertEquals(1,person.personId());
        Assertions.assertEquals("Joe Smith",person.name());
        Assertions.assertEquals(0,person.managerId());
        Assertions.assertNull(person.manager());
        Assertions.assertEquals("Developer",person.role());
        Assertions.assertEquals(13,person.grade());

        person = people.get(1);
        Assertions.assertEquals(2,person.personId());
        Assertions.assertEquals("Jacqueline Ross",person.name());
        Assertions.assertEquals(1,person.managerId());
        Assertions.assertEquals("Joe Smith",person.manager());
        Assertions.assertEquals("Developer",person.role());
        Assertions.assertEquals(12,person.grade());
    }

    @Test
    public void TestRecordQueries_SerialisePeopleAsJSONArray() {
        List<Person> people = RecordQueries.GetPeople();
        Assertions.assertEquals(15, people.size());
        Assertions.assertEquals("[2, \"Jacqueline Ross\", 1, \"Joe Smith\", \"Developer\", 12]",
                people.get(1).toJavaScriptArrayString());
    }
}
