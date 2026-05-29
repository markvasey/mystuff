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
public class AdzunaClient implements JobSourceClient {

    private static final Logger log = LoggerFactory.getLogger(AdzunaClient.class);
    private final WebClient webClient;

    @Value("${adzuna.app-id:}")
    private String appId;

    @Value("${adzuna.app-key:}")
    private String appKey;

    private final String baseUrl;

    public AdzunaClient(WebClient.Builder webClientBuilder, @Value("${adzuna.base-url:https://api.adzuna.com/v1/api/jobs/gb/search/1}") String baseUrl) {
        this.baseUrl = baseUrl;
        this.webClient = webClientBuilder.baseUrl(baseUrl).build();
    }

    @Override
    public Flux<JobListing> fetchJobs(SearchCriteria criteria) {
        if (appId == null || appId.isEmpty() || appKey == null || appKey.isEmpty()) {
            log.warn("Adzuna API credentials not configured. appId: {}, appKey length: {}. Skipping.", appId, appKey != null ? appKey.length() : 0);
            return Flux.empty();
        }

        log.info("Fetching jobs from Adzuna for {} in {}...", criteria.getKeywords(), criteria.getTown());

        return webClient.get()
                .uri(uriBuilder -> {
                    uriBuilder
                        .queryParam("app_id", appId)
                        .queryParam("app_key", appKey)
                        .queryParam("what", criteria.getKeywords())
                        .queryParam("where", criteria.getTown())
                        .queryParam("distance", criteria.getRadius())
                        .queryParam("sort_by", "date")
                        .queryParam("salary_include_unknown", 1)
                        .queryParam("content-type", "application/json");
                    
                    if (criteria.getCategory() != null && !criteria.getCategory().isEmpty()) {
                        uriBuilder.queryParam("category", criteria.getCategory());
                    }
                    
                    if (criteria.isPartTime()) {
                        uriBuilder.queryParam("part_time", 1);
                    }
                    
                    return uriBuilder.build();
                })
                .retrieve()
                .bodyToMono(Map.class)
                .flatMapIterable(response -> {
                    List<Map<String, Object>> results = (List<Map<String, Object>>) response.get("results");
                    return results != null ? results : List.of();
                })
                .map(this::mapToJobListing);
    }

    private JobListing mapToJobListing(Map<String, Object> adzunaJob) {
        JobListing job = new JobListing();
        job.setExternalId(adzunaJob.get("id").toString());
        job.setSource(getSourceName());
        job.setTitle(adzunaJob.get("title").toString());
        job.setCompany(((Map<String, Object>) adzunaJob.get("company")).get("display_name").toString());
        job.setLocation(((Map<String, Object>) adzunaJob.get("location")).get("display_name").toString());
        job.setDescription(adzunaJob.get("description").toString());
        job.setUrl(adzunaJob.get("redirect_url").toString());
        job.setSalaryInfo(adzunaJob.containsKey("salary_min") ? adzunaJob.get("salary_min").toString() : "N/A");
        // Adzuna date format: 2026-05-29T12:00:00Z
        job.setPostedAt(LocalDateTime.now()); // Simplified for now
        return job;
    }

    @Override
    public String getSourceName() {
        return "Adzuna";
    }
}
