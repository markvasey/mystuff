package com.example.jobsearch.client;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.SearchCriteria;
import reactor.core.publisher.Flux;

public interface JobSourceClient {
    Flux<JobListing> fetchJobs(SearchCriteria criteria);
    String getSourceName();
}
