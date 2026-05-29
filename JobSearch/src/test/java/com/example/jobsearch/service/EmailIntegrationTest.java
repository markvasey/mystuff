package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.util.ReflectionTestUtils;

@SpringBootTest
@ActiveProfiles("local")
class EmailIntegrationTest {

    @Autowired
    private EmailService emailService;

    @Test
    void testSendRealEmail() {
        JobListing testJob = new JobListing();
        testJob.setTitle("Test Job for Winchester");
        testJob.setCompany("Test Company");
        testJob.setLocation("Winchester");
        testJob.setRelevanceScore(95);
        testJob.setMatchReason("This is a automated test of the JobSearch email system.");
        testJob.setUrl("http://localhost:8084");

        // Override the recipient specifically for this test
        ReflectionTestUtils.setField(emailService, "toEmail", "markdvasey@icloud.com");

        System.out.println("Attempting to send test email to markdvasey@icloud.com...");
        emailService.sendJobNotification(testJob);
        System.out.println("Test execution complete. Check the logs/inbox.");
    }
}
