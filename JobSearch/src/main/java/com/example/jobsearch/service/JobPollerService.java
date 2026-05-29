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
import org.springframework.transaction.annotation.Transactional;
import java.time.LocalDateTime;
import java.util.List;

@Service
public class JobPollerService {

    private static final Logger log = LoggerFactory.getLogger(JobPollerService.class);

    private final SearchCriteriaRepository criteriaRepository;
    private final JobListingRepository jobListingRepository;
    private final List<JobSourceClient> jobSourceClients;
    private final RelevanceScorerService relevanceScorerService;

    private static final List<String> BLACKLIST = List.of("Software", "Developer", "Engineer", "Programmer", "DevOps", "Data Scientist");
    private static final List<String> LOCATION_BLACKLIST = List.of("Southampton", "Basingstoke", "Portsmouth", "Bournemouth", "Reading", "Andover");

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
    @Transactional
    public void pollJobs() {
        log.info("Starting job poll...");
        List<SearchCriteria> activeCriteria = criteriaRepository.findByActiveTrue();
        
        for (SearchCriteria criteria : activeCriteria) {
            log.info("Polling for town: {}, keywords: {}", criteria.getTown(), criteria.getKeywords());
            
            for (JobSourceClient client : jobSourceClients) {
                List<JobListing> jobs = client.fetchJobs(criteria).collectList().block();
                if (jobs != null) {
                    for (JobListing job : jobs) {
                        // Global Title Blacklist check
                        boolean blacklisted = BLACKLIST.stream().anyMatch(word -> job.getTitle().toLowerCase().contains(word.toLowerCase()));
                        
                        if (blacklisted) {
                            log.info("Skipping blacklisted job title: {}", job.getTitle());
                            continue;
                        }

                        // Location Blacklist check
                        boolean farAway = LOCATION_BLACKLIST.stream().anyMatch(loc -> job.getLocation().toLowerCase().contains(loc.toLowerCase()));
                        if (farAway) {
                            log.info("Skipping job too far away: {} in {}", job.getTitle(), job.getLocation());
                            continue;
                        }

                        if (jobListingRepository.findByExternalIdAndSource(job.getExternalId(), job.getSource()).isEmpty()) {
                            log.info("New job found: {} at {} ({})", job.getTitle(), job.getCompany(), job.getSource());
                            
                            // Track which search town found this job
                            job.setTown(criteria.getTown());

                            // Score the job before saving
                            relevanceScorerService.scoreJob(job);
                            
                            jobListingRepository.save(job);
                        }
                    }
                }
            }
            
            criteria.setLastPolledAt(LocalDateTime.now());
            criteriaRepository.save(criteria);
        }
        log.info("Job poll complete.");
    }
}
