package com.example.jobsearch.client;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.SearchCriteria;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import java.time.LocalDateTime;

//@Component
public class MockJobClient implements JobSourceClient {

    @Override
    public Flux<JobListing> fetchJobs(SearchCriteria criteria) {
        JobListing job = new JobListing();
        job.setExternalId("mock-" + System.currentTimeMillis());
        job.setSource(getSourceName());
        job.setTitle("Software Engineer in " + criteria.getTown());
        job.setCompany("Mock Corp");
        job.setLocation(criteria.getTown());
        job.setDescription("A great job searching for " + criteria.getKeywords());
        job.setUrl("http://example.com/mock-job");
        job.setPostedAt(LocalDateTime.now());
        
        return Flux.just(job);
    }

    @Override
    public String getSourceName() {
        return "MockSource";
    }
}
