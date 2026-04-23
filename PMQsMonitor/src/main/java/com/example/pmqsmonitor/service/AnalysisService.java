package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.AnalysisResultRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Service
public class AnalysisService {

    private static final Logger log = LoggerFactory.getLogger(AnalysisService.class);
    private final WebClient webClient;
    private final String apiKey;
    private final AnalysisResultRepository analysisResultRepository;
    private final ObjectMapper objectMapper;

    public AnalysisService(@Value("${spring.ai.google.genai.api-key}") String apiKey,
                           AnalysisResultRepository analysisResultRepository) {
        this.webClient = WebClient.builder().build();
        this.apiKey = apiKey;
        this.analysisResultRepository = analysisResultRepository;
        this.objectMapper = new ObjectMapper()
                .configure(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    }

    private String executeWithRetry(String url, Map<String, Object> requestBody) {
        int maxAttempts = 5;
        int delayMs = 5000;
        Exception lastException = null;

        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                return webClient.post()
                        .uri(url)
                        .bodyValue(requestBody)
                        .retrieve()
                        .bodyToMono(String.class)
                        .block();
            } catch (Exception e) {
                lastException = e;
                log.warn("Gemini API attempt {} failed: {}. Retrying in {}ms...", attempt, e.getMessage(), delayMs);
                if (attempt < maxAttempts) {
                    try { Thread.sleep(delayMs); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); break; }
                }
            }
        }
        log.error("Gemini API failed after {} attempts. Final error: {}", maxAttempts, lastException != null ? lastException.getMessage() : "Unknown");
        return null;
    }

    public void analyzeUtterance(Utterance question, Utterance answer) {
        if (answer.getAnalysisResult() != null) {
            return;
        }

        String qName = question.getSpeakerName() != null ? question.getSpeakerName() : "Unknown MP";
        String qText = question.getText() != null ? question.getText() : "[No question text available]";
        String aName = answer.getSpeakerName() != null ? answer.getSpeakerName() : "The Respondent";
        String aText = answer.getText() != null ? answer.getText() : "[No answer text available]";

        String prompt = String.format("""
                Analyze the following Prime Minister's Questions (PMQs) exchange.
                Focus specifically on the completeness and relevance of the answer given by %s.
                
                Question from %s: %s
                Answer from %s: %s
                
                Provide your analysis in JSON format with exactly these fields:
                - sentiment: String (e.g., "Defensive", "Honest", "Combative")
                - tone: String (brief description of the mood)
                - completeness: Integer (0-100)
                - relevance: Integer (0-100)
                - isDirectAnswer: Boolean
                - diversionTactics: List of Strings
                - pointsAnswered: List of Strings
                - pointsMissed: List of Strings
                - rational: String (detailed explanation for your scores)
                
                Return ONLY the JSON object.
                """, aName, qName, qText, aName, aText);

        String url = String.format("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=%s", apiKey);

        Map<String, Object> requestBody = Map.of(
            "contents", List.of(Map.of(
                "parts", List.of(Map.of("text", prompt))
            )),
            "generationConfig", Map.of(
                "response_mime_type", "application/json"
            )
        );

        try {
            String responseJson = executeWithRetry(url, requestBody);

            if (responseJson != null) {
                JsonNode root = objectMapper.readTree(responseJson);
                JsonNode candidates = root.path("candidates");
                if (!candidates.isMissingNode() && candidates.size() > 0) {
                    JsonNode parts = candidates.get(0).path("content").path("parts");
                    if (!parts.isMissingNode() && parts.size() > 0) {
                        String innerJson = parts.get(0).path("text").asText();
                        AnalysisResult result = objectMapper.readValue(innerJson, AnalysisResult.class);

                        if (result != null) {
                            result.setUtterance(answer);
                            result.setAnalyzedAt(LocalDateTime.now());
                            
                            if (answer.getId() != null) {
                                analysisResultRepository.save(result);
                                log.info("Analysis saved successfully for {}", answer.getExternalId());
                            } else {
                                log.info("Analysis generated successfully (skipping DB save for transient utterance)");
                                answer.setAnalysisResult(result);
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            log.error("Gemini Parsing Error: {}", e.getMessage(), e);
        }
    }

    public String summarizeRationales(List<String> rationales) {
        if (rationales == null || rationales.isEmpty()) {
            return "No analysis data available to summarize.";
        }

        String combinedRationales = String.join("\n---\n", rationales);
        String summaryPrompt = String.format("""
                You are a senior political editor. Below is a list of AI-generated rationales analyzing individual answers from a Prime Minister's Questions (PMQs) session.
                
                Please provide a concise "Executive Summary" (MAXIMUM 200 WORDS) of the Prime Minister's performance in this session based on these analyses. 
                Focus on recurring themes, the overall tone (e.g., was it a defensive or confident session?), and identify if there were specific topics where the PM was particularly evasive or informative.
                
                RATIONALES TO SUMMARIZE:
                %s
                """, combinedRationales);

        Map<String, Object> requestBody = Map.of(
            "contents", List.of(Map.of(
                "parts", List.of(Map.of("text", summaryPrompt))
            ))
        );

        String url = String.format("https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=%s", apiKey);
        
        String responseJson = executeWithRetry(url, requestBody);

        if (responseJson != null) {
            try {
                JsonNode root = objectMapper.readTree(responseJson);
                JsonNode candidates = root.path("candidates");
                if (!candidates.isMissingNode() && candidates.size() > 0) {
                    return candidates.get(0).path("content").path("parts").get(0).path("text").asText();
                }
            } catch (Exception e) {
                log.error("Summary Parsing Error: {}", e.getMessage());
            }
        }
        return "Failed to generate AI session summary after retries.";
    }
}
