package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import com.example.jobsearch.entity.Person;
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
        Person testPerson = new Person();
        testPerson.setName("Test User");

        JobListing testJob = new JobListing();
        testJob.setTitle("Test Job for Winchester");
        testJob.setCompany("Test Company");
        testJob.setLocation("Winchester");
        testJob.setUrl("http://localhost:8084");

        JobMatch testMatch = new JobMatch();
        testMatch.setPerson(testPerson);
        testMatch.setJobListing(testJob);
        testMatch.setRelevanceScore(95);
        testMatch.setMatchReason("This is a automated test of the JobSearch email system.");
        testMatch.setTown("Winchester");

        // Override the recipient specifically for this test
        ReflectionTestUtils.setField(emailService, "toEmail", "markdvasey@icloud.com");

        System.out.println("Attempting to send test email to markdvasey@icloud.com...");
        emailService.sendJobNotification(testMatch);
        System.out.println("Test execution complete. Check the logs/inbox.");
    }
}
