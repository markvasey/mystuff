package com.example.jobsearch.client;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.SearchCriteria;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@Component
public class ReedClient implements JobSourceClient {

    private static final Logger log = LoggerFactory.getLogger(ReedClient.class);
    private final WebClient webClient;

    @Value("${reed.api-key:}")
    private String apiKey;

    public ReedClient(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("https://www.reed.co.uk/api/1.0").build();
    }

    @Override
    public Flux<JobListing> fetchJobs(SearchCriteria criteria) {
        if (apiKey == null || apiKey.isEmpty()) {
            log.warn("Reed API key not configured. Skipping.");
            return Flux.empty();
        }

        log.info("Fetching jobs from Reed for {} in {}...", criteria.getKeywords(), criteria.getTown());

        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/search")
                        .queryParam("keywords", criteria.getKeywords())
                        .queryParam("locationName", criteria.getTown())
                        .queryParam("distanceFromLocation", criteria.getRadius())
                        .queryParam("partTime", criteria.isPartTime())
                        .build())
                .headers(headers -> headers.setBasicAuth(apiKey, ""))
                .retrieve()
                .bodyToMono(Map.class)
                .flatMapIterable(response -> {
                    List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");
                    int count = results != null ? results.size() : 0;
                    log.info("Reed returned {} results for {} in {}.", count, criteria.getKeywords(), criteria.getTown());
                    return results != null ? results : List.of();
                })
                .map(this::mapToJobListing);
    }

    private JobListing mapToJobListing(Map<String, Object> reedJob) {
        JobListing job = new JobListing();
        job.setExternalId(reedJob.get("jobId").toString());
        job.setSource(getSourceName());
        job.setTitle(reedJob.get("jobTitle").toString());
        job.setCompany(reedJob.get("employerName").toString());
        job.setLocation(reedJob.get("locationName").toString());
        job.setDescription(reedJob.get("jobDescription").toString());
        job.setUrl(reedJob.get("jobUrl").toString());
        
        Object minSalary = reedJob.get("minimumSalary");
        job.setSalaryInfo(minSalary != null ? "£" + minSalary.toString() : "N/A");
        
        job.setPostedAt(LocalDateTime.now()); // Reed provides date in DD/MM/YYYY usually, simplified here
        return job;
    }

    @Override
    public String getSourceName() {
        return "Reed";
    }
}
