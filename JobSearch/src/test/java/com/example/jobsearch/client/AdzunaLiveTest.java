package com.example.jobsearch.client;

import com.example.jobsearch.entity.SearchCriteria;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import reactor.test.StepVerifier;

@SpringBootTest
@ActiveProfiles("test") // Use H2 for DB, but keys will load from application.properties
class AdzunaLiveTest {

    @Autowired
    private AdzunaClient adzunaClient;

    @Test
    void testLiveFetchFromAdzuna() {
        SearchCriteria criteria = new SearchCriteria();
        criteria.setTown("Winchester");
        criteria.setKeywords("Software");

        adzunaClient.fetchJobs(criteria)
                .take(1) 
                .as(StepVerifier::create)
                .expectNextCount(1)
                .verifyComplete();
    }
}
