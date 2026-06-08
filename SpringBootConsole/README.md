# SpringBootConsole

A Spring Boot application with a console runner and a web controller.

## Features
- **Case-insensitive URL matching:** URLs like `/hello`, `/HELLO`, and `/HeLlO` all map to the same endpoint.
- **Custom Scripts:** Dedicated scripts for starting, stopping, and verifying the application.

## Endpoints
- **Hello World:** `http://localhost:8081/hello`

## How to Run
Use the scripts in the `scripts/` directory to manage the application:

- **Start:** `./scripts/start_app.sh`
- **Stop:** `./scripts/stop_app.sh`
- **Verify Scripts:** `./scripts/verify_scripts.sh`

## Running Tests
To run the Maven unit tests:
```bash
./mvnw test
```
The project includes `HelloControllerTest.java` which verifies case-insensitive URL matching.

## Configuration
- **Port:** 8081 (defined in `src/main/resources/application.properties`)
- **Java Version:** 26.0.1 (Temurin)
