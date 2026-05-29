package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
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

    public void sendJobNotification(JobListing job) {
        if (fromEmail.isEmpty() || toEmail.isEmpty() || fromEmail.contains("YOUR_YAHOO")) {
            log.warn("Email not configured. Skipping.");
            return;
        }

        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(fromEmail);
            message.setTo(toEmail);
            message.setSubject("New Job Match: " + job.getTitle() + " (" + job.getRelevanceScore() + "%)");
            
            String body = String.format("""
                Hi Maya,
                
                I found a new job listing for you in Winchester:
                
                Title: %s
                Company: %s
                Location: %s
                Match Score: %d%%
                
                AI INSIGHT:
                %s
                
                VIEW JOB:
                %s
                
                Good luck!
                """, 
                job.getTitle(), 
                job.getCompany(), 
                job.getLocation(), 
                job.getRelevanceScore(),
                job.getMatchReason(),
                job.getUrl());
            
            message.setText(body);
            mailSender.send(message);
            log.info("Email sent successfully for job: {}", job.getTitle());
        } catch (Exception e) {
            log.error("Failed to send email for job {}: {}", job.getTitle(), e.getMessage());
        }
    }
}
