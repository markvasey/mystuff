package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

@Service
public class RelevanceScorerService {

    private static final Logger log = LoggerFactory.getLogger(RelevanceScorerService.class);
    private final ChatClient chatClient;
    private String resumeContent;

    @Value("${app.resume-path:MayaResume.txt}")
    private String resumePath;

    public RelevanceScorerService(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    private synchronized void loadResume() {
        if (resumeContent == null) {
            try {
                resumeContent = Files.readString(Path.of(resumePath));
                log.info("Loaded resume from {}", resumePath);
            } catch (Exception e) {
                log.error("Failed to load resume: {}", e.getMessage());
                resumeContent = "Resume not found.";
            }
        }
    }

    public void scoreJob(JobListing job) {
        loadResume();
        
        String prompt = String.format("""
            You are a career advisor for Maya Vasey. 
            Evaluate the following job listing against her resume and provide a relevance score (0-100) and a brief reason.
            
            MAYA'S RESUME:
            %s
            
            JOB LISTING:
            Title: %s
            Company: %s
            Location: %s
            Description: %s
            
            Output format:
            SCORE: [number]
            REASON: [one sentence summary of fit]
            """, resumeContent, job.getTitle(), job.getCompany(), job.getLocation(), job.getDescription());

        try {
            String response = chatClient.prompt(prompt).call().content();
            log.debug("AI Response for {}: {}", job.getTitle(), response);
            
            parseResponse(job, response);
        } catch (Exception e) {
            log.error("AI scoring failed for {}: {}", job.getTitle(), e.getMessage());
            job.setRelevanceScore(0);
            job.setMatchReason("AI analysis unavailable.");
        }
    }

    private void parseResponse(JobListing job, String response) {
        Pattern scorePattern = Pattern.compile("SCORE:\\s*(\\d+)");
        Pattern reasonPattern = Pattern.compile("REASON:\\s*(.*)", Pattern.DOTALL);

        Matcher scoreMatcher = scorePattern.matcher(response);
        if (scoreMatcher.find()) {
            job.setRelevanceScore(Integer.parseInt(scoreMatcher.group(1).trim()));
        }

        Matcher reasonMatcher = reasonPattern.matcher(response);
        if (reasonMatcher.find()) {
            job.setMatchReason(reasonMatcher.group(1).trim());
        }
    }
}
