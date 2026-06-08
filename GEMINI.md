## Workspace & Security
- **Strict Read-Only Access:** All folders and files under `/home/markvasey/Dropbox/` are strictly **READ-ONLY**, with the sole exception of the `/home/markvasey/Dropbox/GitHub/` directory.
- **Modification Policy:** Gemini is ONLY permitted to modify, create, or delete files within the `GitHub` directory and its subdirectories.
- **Data Integrity:** Never attempt to write to, update, or move files in any other part of the Dropbox folder (e.g., `Data/`, `Drawings/`, etc.) under any circumstances.

## Java Environment
- **JDK Version:** OpenJDK 26.0.1 (Temurin)
- **JDK Path:** `/home/markvasey/.sdkman/candidates/java/26.0.1-tem`
- Always ensure `JAVA_HOME` is set to this path and included in the `PATH` when running build or test commands.

## Build Tools
- **Maven:** Use the Maven wrapper (`./mvnw`) whenever available in a project directory.
- Avoid using the system `mvn` command as it may not be installed or configured correctly.

## Project Specifics
### SpringBootConsole
- **Server Port:** Defaults to `8081` (configured in `application.properties`).
- **URL Matching:** Configured for case-insensitive URL matching via `WebConfig.java` using `AntPathMatcher`.

## General Engineering Standards
- Adhere to Spring Boot 3.4.3 conventions and Java 17+ features.
- **Testing:** ALWAYS add a new test case to the existing test file (if one exists) or create a new test file to verify every new piece of functionality.
- Ensure all new controllers or endpoints are verified with both exact and case-insensitive URL tests.
