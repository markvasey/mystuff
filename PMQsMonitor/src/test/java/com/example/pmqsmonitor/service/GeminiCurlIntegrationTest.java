package com.example.pmqsmonitor.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
@ActiveProfiles("local")
class GeminiCurlIntegrationTest {

    @Value("${spring.ai.google.genai.api-key}")
    private String apiKey;

    @Test
    void testSuccessfulCurlMirror() {
        // This test mirrors the successful curl call:
        // curl -s -X POST "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=..."
        
        WebClient webClient = WebClient.builder().build();
        
        String url = String.format("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=%s", apiKey);
        
        Map<String, Object> requestBody = Map.of(
            "contents", List.of(Map.of(
                "parts", List.of(Map.of("text", "Hello, are you working?"))
            ))
        );

        System.out.println("Executing mirrored curl call to: " + url.replace(apiKey, "REDACTED"));

        String responseJson = webClient.post()
                .uri(url)
                .bodyValue(requestBody)
                .retrieve()
                .bodyToMono(String.class)
                .block();

        assertNotNull(responseJson);
        System.out.println("Response received: " + responseJson);
        assertTrue(responseJson.contains("candidates"), "Response should contain candidates");
    }
}
