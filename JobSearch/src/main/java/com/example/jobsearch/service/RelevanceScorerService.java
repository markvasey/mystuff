package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
public class RelevanceScorerService {

    private static final Logger log = LoggerFactory.getLogger(RelevanceScorerService.class);
    private final ChatClient chatClient;
    private final Map<String, String> resumeCache = new ConcurrentHashMap<>();

    public RelevanceScorerService(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    private String getResumeContent(String resumePath) {
        return resumeCache.computeIfAbsent(resumePath, path -> {
            try {
                String content = Files.readString(Path.of(path));
                log.info("Loaded resume from {}", path);
                return content;
            } catch (Exception e) {
                log.error("Failed to load resume {}: {}", path, e.getMessage());
                return "Resume not found.";
            }
        });
    }

    public void scoreJob(JobMatch match) {
        JobListing job = match.getJobListing();
        String personName = match.getPerson().getName();
        String resumePath = match.getPerson().getResumePath();
        String resumeContent = getResumeContent(resumePath);
        
        String prompt = String.format("""
            You are a career advisor for %s. 
            Evaluate the following job listing against their resume and provide a relevance score (0-100) and a brief reason.
            
            %s'S RESUME:
            %s
            
            JOB LISTING:
            Title: %s
            Company: %s
            Location: %s
            Description: %s
            
            Output format:
            SCORE: [number]
            REASON: [one sentence summary of fit]
            """, personName, personName.toUpperCase(), resumeContent, job.getTitle(), job.getCompany(), job.getLocation(), job.getDescription());

        try {
            String response = chatClient.prompt(prompt).call().content();
            log.debug("AI Response for {}: {}", job.getTitle(), response);
            
            parseResponse(match, response);
        } catch (Exception e) {
            log.error("AI scoring failed for {}: {}", job.getTitle(), e.getMessage());
            match.setRelevanceScore(0);
            match.setMatchReason("AI analysis unavailable.");
        }
    }

    private void parseResponse(JobMatch match, String response) {
        Pattern scorePattern = Pattern.compile("SCORE:\\s*(\\d+)");
        Pattern reasonPattern = Pattern.compile("REASON:\\s*(.*)", Pattern.DOTALL);

        Matcher scoreMatcher = scorePattern.matcher(response);
        if (scoreMatcher.find()) {
            match.setRelevanceScore(Integer.parseInt(scoreMatcher.group(1).trim()));
        }

        Matcher reasonMatcher = reasonPattern.matcher(response);
        if (reasonMatcher.find()) {
            match.setMatchReason(reasonMatcher.group(1).trim());
        }
    }
}
