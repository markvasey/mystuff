package com.example.jobsearch.client;

import com.example.jobsearch.entity.SearchCriteria;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import reactor.test.StepVerifier;

@SpringBootTest
@ActiveProfiles("local") // Use local to get real keys for live test
class AdzunaLiveTest {

    @Autowired
    private AdzunaClient adzunaClient;

    @Test
    void testLiveFetchFromAdzuna() {
        SearchCriteria criteria = new SearchCriteria();
        criteria.setTown("London");
        criteria.setKeywords("Staff");

        adzunaClient.fetchJobs(criteria)
                .take(1) 
                .as(StepVerifier::create)
                .expectNextCount(1)
                .verifyComplete();
    }
}
