package records;

public record Person(int personId,
                     String name,
                     Integer managerId,
                     String manager,
                     String role,
                     int grade
) {
}
