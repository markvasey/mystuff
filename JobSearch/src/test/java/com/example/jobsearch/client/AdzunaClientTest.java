package com.example.jobsearch.client;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.SearchCriteria;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.test.StepVerifier;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AdzunaClientTest {

    private MockWebServer mockWebServer;
    private AdzunaClient adzunaClient;

    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();

        WebClient.Builder webClientBuilder = WebClient.builder();
        adzunaClient = new AdzunaClient(webClientBuilder, mockWebServer.url("/").toString());
        
        // Inject keys via reflection for testing
        ReflectionTestUtils.setField(adzunaClient, "appId", "test-id");
        ReflectionTestUtils.setField(adzunaClient, "appKey", "test-key");
    }

    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }

    @Test
    void testFetchJobsParsing() {
        String mockResponse = """
        {
          "results": [
            {
              "id": "12345",
              "title": "Software Engineer",
              "company": { "display_name": "Test Co" },
              "location": { "display_name": "Winchester" },
              "description": "Job description here",
              "redirect_url": "http://example.com",
              "salary_min": 50000
            }
          ]
        }
        """;

        mockWebServer.enqueue(new MockResponse()
                .setBody(mockResponse)
                .addHeader("Content-Type", "application/json"));

        SearchCriteria criteria = new SearchCriteria();
        criteria.setTown("Winchester");
        criteria.setKeywords("Java");

        StepVerifier.create(adzunaClient.fetchJobs(criteria))
                .assertNext(job -> {
                    assertEquals("12345", job.getExternalId());
                    assertEquals("Software Engineer", job.getTitle());
                    assertEquals("Test Co", job.getCompany());
                    assertEquals("Winchester", job.getLocation());
                    assertEquals("Adzuna", job.getSource());
                })
                .verifyComplete();
    }
}
