package com.example.jobsearch.client;

import com.example.jobsearch.entity.SearchCriteria;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import reactor.test.StepVerifier;

@SpringBootTest
@ActiveProfiles("local")
class ReedLiveTest {

    @Autowired
    private ReedClient reedClient;

    @Test
    void testLiveFetchFromReed() {
        SearchCriteria criteria = new SearchCriteria();
        criteria.setTown("London");
        criteria.setKeywords("Staff");
        criteria.setRadius(10);
        criteria.setPartTime(false);

        // We verify that the API responds and returns at least one result.
        reedClient.fetchJobs(criteria)
                .take(1)
                .as(StepVerifier::create)
                .expectNextCount(1)
                .verifyComplete();
    }
}
