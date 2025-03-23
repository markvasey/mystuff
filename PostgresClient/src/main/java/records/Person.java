package records;

public record Person(int personId,
                     String name,
                     Integer managerId,
                     String manager,
                     String role,
                     int grade
) {
    public String toJavaScriptArrayString() {
        return "[" +
                personId + ", " +
                "\"" + name + "\", " +
                managerId + ", " +
                "\"" + manager + "\", " +
                "\"" + role + "\", " +
                grade +
                "]";
    }


}
