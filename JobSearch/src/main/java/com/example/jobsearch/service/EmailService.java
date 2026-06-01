package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import com.example.jobsearch.entity.Person;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class EmailService {

    private static final Logger log = LoggerFactory.getLogger(EmailService.class);
    private final JavaMailSender mailSender;

    @Value("${spring.mail.username:}")
    private String fromEmail;

    @Value("${app.mail.to:}")
    private String toEmail;

    public EmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    public void sendJobNotification(JobMatch match) {
        if (fromEmail.isEmpty() || fromEmail.contains("YOUR_YAHOO")) {
            log.warn("Email sender not configured. Skipping.");
            return;
        }

        JobListing job = match.getJobListing();
        Person person = match.getPerson();
        String personName = person.getName();
        String recipientEmail = person.getEmail() != null ? person.getEmail() : toEmail;

        if (recipientEmail == null || recipientEmail.isEmpty()) {
            log.warn("No recipient email found for {}. Skipping.", personName);
            return;
        }

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(recipientEmail);
            
            String scoreText = match.getRelevanceScore() != null ? match.getRelevanceScore() + "%" : "Possible Match";
            message.setSubject("New Job Match for " + personName + ": " + job.getTitle() + " (" + scoreText + ")");
            
            String body = String.format("""
                Hi %s,
                
                I found a new job listing for you in %s:
                
                Title: %s
                Company: %s
                Location: %s
                Match Score: %s
                
                AI INSIGHT:
                %s
                
                VIEW JOB:
                %s
                
                Good luck!
                """, 
                personName,
                match.getTown(),
                job.getTitle(), 
                job.getCompany(), 
                job.getLocation(), 
                scoreText,
                match.getMatchReason() != null ? match.getMatchReason() : "None",
                job.getUrl());
            
            message.setText(body);
            mailSender.send(message);
            log.info("Email sent successfully for job: {}", job.getTitle());
        } catch (Exception e) {
            log.error("Failed to send email for job {}: {}", job.getTitle(), e.getMessage());
        }
    }
}
