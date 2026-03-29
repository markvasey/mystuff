---
name: spring-boot-web-app
description: "Create and configure a new Spring Boot 3.x web application with specialized settings like case-insensitive URLs and local execution scripts."
---

# Spring Boot Web App Creation

This skill helps you bootstrap a Spring Boot 3.4.3 project using Java 23 and Maven, with a focus on case-insensitive URL matching and local lifecycle management.

## Project Structure
- **Source:** Uses standard Maven project structure (`src/main/java`).
- **Scripts:** Includes `scripts/` for starting, stopping, and verifying the app.

## Case-Insensitive URL Matching
To enable case-insensitive URLs, create a `WebConfig.java` in your `org.example` package (or equivalent) using the following pattern:

```java
package org.example;

import org.springframework.context.annotation.Configuration;
import org.springframework.util.AntPathMatcher;
import org.springframework.web.servlet.config.annotation.PathMatchConfigurer;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void configurePathMatch(PathMatchConfigurer configurer) {
        AntPathMatcher matcher = new AntPathMatcher();
        matcher.setCaseSensitive(false);
        configurer.setPathMatcher(matcher);
    }
}
```

## Java Environment
- **JAVA_HOME:** `/home/markvasey/.jdks/openjdk-23.0.1`
- **Maven:** Use `./mvnw` whenever available.

## Workflow
1.  **Initialize Project:** Run `mvn archetype:generate` or copy a boilerplate `pom.xml`.
2.  **Add Configuration:** Create `WebConfig.java` to enable case-insensitive matching.
3.  **Add Scripts:** Use the templates provided in this skill's `assets/scripts/` folder to manage the application lifecycle.
4.  **Verify:** Run `scripts/verify_scripts.sh` to ensure everything is set up correctly.
