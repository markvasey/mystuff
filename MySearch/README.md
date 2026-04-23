# MySearch - Personal Personal Search Engine

MySearch is a Spring Boot application designed to aggregate personal data from various sources (Yahoo Mail, Dropbox, Evernote) into a local, searchable PostgreSQL database. It provides a familiar "Google-like" Web UI and exposes its search capabilities as a **Model Context Protocol (MCP)** server for AI assistants like Gemini.

## 🚀 Features

- **Personal Data Aggregation:** Automatically syncs and indexes data from multiple personal accounts.
- **Full-Text Search:** Powered by PostgreSQL's `tsvector` indexing for fast, complex queries.
- **Google-Style Web UI:** A clean, centered search interface with ranked results and detailed snippets.
- **MCP Server Integration:** Exposes 4 specialized tools to AI assistants:
  - `searchByText(query)`: General PostgreSQL Full-Text Search.
  - `searchByDateRange(start, end)`: Filter items within specific ISO dates.
  - `listBySource(source)`: Filter items by source (e.g., `YAHOO_MAIL`).
  - `getItemDetails(uuid)`: Retrieve full content and metadata for a specific item.
- **Incremental Syncing:** Uses smart watermarks (IMAP UIDs, Dropbox Cursors, Evernote Timestamps) to fetch only new data.

## 🛠️ Technology Stack

- **Backend:** Java 23, Spring Boot 3.4.3, Spring Data JPA.
- **Database:** PostgreSQL (with Full-Text Search).
- **Frontend:** Thymeleaf + Tailwind CSS.
- **Integrations:** Jakarta Mail (IMAP), Dropbox SDK, Evernote SDK.
- **AI Integration:** Spring AI MCP Server (stdio transport).

## ⚙️ Configuration

The application uses Spring Profiles. Most sensitive configuration should be placed in `src/main/resources/application-local.properties`.

### Database
```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/mysearch
spring.datasource.username=postgres
spring.datasource.password=your_password
```

### Source API Credentials
```properties
# Yahoo Mail (IMAP with App Password)
yahoo.mail.username=user@ymail.com
yahoo.mail.app-password=your_app_password

# Dropbox (Long-lived Access Token)
dropbox.access-token=your_token

# Evernote (Developer Token)
evernote.access-token=your_token
```

### MCP Mode Tuning
To ensure a stable connection with MCP clients (which use `stdio`), the application is configured to suppress "noisy" output:
```properties
# Suppress banners and console logging during MCP sessions
spring.main.banner-mode=off
logging.pattern.console=
logging.file.name=/tmp/mysearch-mcp.log
```

### To Run as a web server

Delete these lines from application-local.properties

spring.main.banner-mode=off
logging.pattern.console=
logging.file.name=/tmp/mysearch-mcp.log
spring.main.web-application-type=none

Change the port in application.properties to not clash with the MCP Server 

## 🏃 Running the Application

### Web Mode
1. Ensure PostgreSQL is running.
2. Build the app: `./mvnw clean package -DskipTests`
3. Run: `java -jar target/mysearch-0.0.1-SNAPSHOT.jar`
4. Access at: `http://localhost:8081`

### MCP Mode
To use MySearch with an MCP client (like Gemini CLI), add the following to your `.gemini/settings.json`:

```json
{
  "mcpServers": {
    "my-search": {
      "command": "java",
      "args": [
        "-Dspring.profiles.active=local",
        "-jar",
        "/path/to/MySearch/target/mysearch-0.0.1-SNAPSHOT.jar",
        "--spring.main.banner-mode=off",
        "--logging.level.root=WARN"
      ]
    }
  }
}
```

> **Note:** For MCP `stdio` transport to work reliably, the web server is currently disabled in `application-local.properties` via `spring.main.web-application-type=none`. To use the Web UI again, change this to `servlet` and rebuild.

## 🔍 Search Syntax
PostgreSQL FTS supports:
- **Phrases:** `"mortgage approval"`
- **Negation:** `mortgage -denied`
- **OR Queries:** `mortgage OR loan`

## 🧪 Testing
Run the test suite using Maven:
```bash
./mvnw test
```
The project follows a "test-first" standard for all new controllers and sync logic.
