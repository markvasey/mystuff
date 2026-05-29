package com.example.jobsearch.service;

import com.example.jobsearch.client.JobSourceClient;
import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.JobListingRepository;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import java.time.LocalDateTime;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

@Service
public class JobPollerService {

    private static final Logger log = LoggerFactory.getLogger(JobPollerService.class);

    private final SearchCriteriaRepository criteriaRepository;
    private final JobListingRepository jobListingRepository;
    private final List<JobSourceClient> jobSourceClients;
    private final RelevanceScorerService relevanceScorerService;

    private static final List<String> BLACKLIST = List.of("Software", "Developer", "Engineer", "Programmer", "DevOps", "Data Scientist", "Data Analyst", "Technician", "Financial", "Technical", "Bid", "Forklift", "Global Asset Manager", "German", "Construction", "Logistics", "Defence", "Receptionist", "Warehouse");
    private static final List<String> LOCATION_BLACKLIST = List.of("Southampton", "Basingstoke", "Portsmouth", "Bournemouth", "Reading", "Andover");

    private final AtomicBoolean polling = new AtomicBoolean(false);

    public JobPollerService(SearchCriteriaRepository criteriaRepository,
                            JobListingRepository jobListingRepository,
                            List<JobSourceClient> jobSourceClients,
                            RelevanceScorerService relevanceScorerService) {
        this.criteriaRepository = criteriaRepository;
        this.jobListingRepository = jobListingRepository;
        this.jobSourceClients = jobSourceClients;
        this.relevanceScorerService = relevanceScorerService;
    }

    @Scheduled(cron = "${app.polling.cron}")
    public void pollJobs() {
        if (!polling.compareAndSet(false, true)) {
            log.info("Poll already in progress, skipping.");
            return;
        }
        try {
            log.info("Starting job poll...");
            List<SearchCriteria> activeCriteria = criteriaRepository.findByActiveTrue();
            
            for (SearchCriteria criteria : activeCriteria) {
                processCriteria(criteria);
                criteria.setLastPolledAt(LocalDateTime.now());
                criteriaRepository.save(criteria);
            }
            log.info("Job poll complete.");
        } catch (Exception e) {
            log.error("Error during job poll: {}", e.getMessage());
        } finally {
            polling.set(false);
        }
    }

    private void processCriteria(SearchCriteria criteria) {
        log.info("Polling for town: {}, keywords: {}", criteria.getTown(), criteria.getKeywords());
        
        for (JobSourceClient client : jobSourceClients) {
            try {
                List<JobListing> jobs = client.fetchJobs(criteria).collectList().block();
                if (jobs != null) {
                    for (JobListing job : jobs) {
                        if (isBlacklisted(job)) {
                            continue;
                        }

                        if (jobListingRepository.findByExternalIdAndSource(job.getExternalId(), job.getSource()).isEmpty() &&
                            jobListingRepository.findByTitleAndCompanyAndTown(job.getTitle(), job.getCompany(), criteria.getTown()).isEmpty()) {
                            log.info("New job found: {} at {} ({})", job.getTitle(), job.getCompany(), job.getSource());
                            job.setTown(criteria.getTown());
                            relevanceScorerService.scoreJob(job);
                            
                            // Auto-archive if score is 0
                            if (job.getRelevanceScore() != null && job.getRelevanceScore() == 0) {
                                log.info("Auto-archiving job with 0% match: {}", job.getTitle());
                                job.setStatus("ARCHIVED");
                            }
                            
                            jobListingRepository.save(job);
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Error fetching jobs from {} for {}: {}", client.getSourceName(), criteria.getKeywords(), e.getMessage());
            }
        }
    }

    private boolean isBlacklisted(JobListing job) {
        boolean titleBlacklisted = BLACKLIST.stream().anyMatch(word -> job.getTitle().toLowerCase().contains(word.toLowerCase()));
        if (titleBlacklisted) {
            log.info("Skipping blacklisted job title: {}", job.getTitle());
            return true;
        }

        boolean locationBlacklisted = LOCATION_BLACKLIST.stream().anyMatch(loc -> job.getLocation().toLowerCase().contains(loc.toLowerCase()));
        if (locationBlacklisted) {
            log.info("Skipping job too far away: {} in {}", job.getTitle(), job.getLocation());
            return true;
        }
        return false;
    }

    public boolean isPolling() {
        return polling.get();
    }
}
