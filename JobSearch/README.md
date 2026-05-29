# JobSearch

JobSearch is a Spring Boot application that monitors job listings in specific towns on a schedule.

## Features
- Multi-site scanning (Adzuna, Reed, etc.)
- Scheduled polling via `@Scheduled`
- Web Dashboard for viewing latest jobs
- Search Criteria management
- Case-insensitive URL matching

## Getting Started

### Prerequisites
- Java 23
- PostgreSQL (configured in `application.properties`)

### Running the App
Use the provided scripts:
- `./scripts/start_app.sh`: Starts the app in the background.
- `./scripts/stop_app.sh`: Stops the background app.

The app runs on port `8084` by default.

### Configuration
Add your API keys to `src/main/resources/application.properties`:
```properties
adzuna.app-id=YOUR_ID
adzuna.app-key=YOUR_KEY
```

## Testing
Run tests with:
```bash
./mvnw test
```
