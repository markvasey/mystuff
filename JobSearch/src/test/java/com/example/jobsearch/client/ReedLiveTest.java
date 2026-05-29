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
    void testLiveFetchFromReedWinchester() {
        SearchCriteria criteria = new SearchCriteria();
        criteria.setTown("Winchester");
        criteria.setKeywords("Retail");
        criteria.setRadius(5);
        criteria.setPartTime(true);

        reedClient.fetchJobs(criteria)
                .as(StepVerifier::create)
                .expectNextCount(1) // Should find at least one retail job in Winchester
                .verifyComplete();
    }
}
