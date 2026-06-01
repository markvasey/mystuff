package com.example.jobsearch.service;

import com.example.jobsearch.client.JobSourceClient;
import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import com.example.jobsearch.entity.Person;
import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.JobListingRepository;
import com.example.jobsearch.repository.JobMatchRepository;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
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
    private final JobMatchRepository jobMatchRepository;
    private final List<JobSourceClient> jobSourceClients;
    private final RelevanceScorerService relevanceScorerService;

    @Value("${app.polling.cron}")
    private String cronExpression;

    private final AtomicBoolean polling = new AtomicBoolean(false);

    public JobPollerService(SearchCriteriaRepository criteriaRepository,
                            JobListingRepository jobListingRepository,
                            JobMatchRepository jobMatchRepository,
                            List<JobSourceClient> jobSourceClients,
                            RelevanceScorerService relevanceScorerService) {
        this.criteriaRepository = criteriaRepository;
        this.jobListingRepository = jobListingRepository;
        this.jobMatchRepository = jobMatchRepository;
        this.jobSourceClients = jobSourceClients;
        this.relevanceScorerService = relevanceScorerService;
    }

    @Scheduled(cron = "${app.polling.cron}")
    public void scheduledPoll() {
        if ("-".equals(cronExpression)) {
            return;
        }
        pollJobs();
    }

    public void pollJobs() {
        if (!polling.compareAndSet(false, true)) {
            log.info("Poll already in progress, skipping.");
            return;
        }
        // Run in a new thread to avoid blocking the controller redirect
        new Thread(() -> {
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
        }).start();
    }

    private void processCriteria(SearchCriteria criteria) {
        Person person = criteria.getPerson();
        if (person == null) {
            log.warn("Skipping criteria {} as it has no associated Person.", criteria.getId());
            return;
        }

        log.info("Polling for {} in town: {}, keywords: {}", person.getName(), criteria.getTown(), criteria.getKeywords());
        
        List<String> blacklist = parseList(person.getBlacklist());
        List<String> locationBlacklist = parseList(person.getLocationBlacklist());
        List<String> possibleList = parseList(person.getPossibleList());

        for (JobSourceClient client : jobSourceClients) {
            try {
                List<JobListing> jobs = client.fetchJobs(criteria).collectList().block();
                if (jobs != null) {
                    for (JobListing jobData : jobs) {
                        if (isBlacklisted(jobData, blacklist, locationBlacklist)) {
                            continue;
                        }

                        // Try to find existing job listing globally
                        JobListing existingJob = jobListingRepository.findByExternalIdAndSource(jobData.getExternalId(), jobData.getSource())
                                .orElseGet(() -> jobListingRepository.findByTitleAndCompanyAndLocation(jobData.getTitle(), jobData.getCompany(), jobData.getLocation())
                                        .stream().findFirst().orElse(null));

                        JobListing job;
                        if (existingJob == null) {
                            log.info("New job found: {} at {} ({})", jobData.getTitle(), jobData.getCompany(), jobData.getSource());
                            job = jobListingRepository.save(jobData);
                        } else {
                            job = existingJob;
                        }

                        // Create JobMatch for this person if it doesn't exist
                        if (jobMatchRepository.findByPersonIdAndJobListingId(person.getId(), job.getId()).isEmpty()) {
                            JobMatch match = new JobMatch();
                            match.setPerson(person);
                            match.setJobListing(job);
                            match.setSearchCriteria(criteria);
                            match.setTown(criteria.getTown());

                            boolean isPossible = possibleList.stream().anyMatch(word -> job.getTitle().toLowerCase().contains(word.toLowerCase().trim()));
                            
                            if (isPossible) {
                                log.info("Marking job match as POSSIBLE (skipping AI): {}", job.getTitle());
                                match.setStatus("POSSIBLE");
                            } else {
                                relevanceScorerService.scoreJob(match);
                                
                                if (match.getRelevanceScore() != null && match.getRelevanceScore() == 0) {
                                    log.info("Auto-archiving job match with 0% score for {}: {}", person.getName(), job.getTitle());
                                    match.setStatus("ARCHIVED");
                                }
                            }
                            
                            jobMatchRepository.save(match);
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Error fetching jobs from {} for {}: {}", client.getSourceName(), criteria.getKeywords(), e.getMessage());
            }
        }
    }

    private List<String> parseList(String raw) {
        if (raw == null || raw.isEmpty()) return List.of();
        return List.of(raw.split(","));
    }

    private boolean isBlacklisted(JobListing job, List<String> blacklist, List<String> locationBlacklist) {
        boolean titleBlacklisted = blacklist.stream().anyMatch(word -> job.getTitle().toLowerCase().contains(word.toLowerCase().trim()));
        if (titleBlacklisted) {
            log.info("Skipping blacklisted job title: {}", job.getTitle());
            return true;
        }

        boolean locationBlacklisted = locationBlacklist.stream().anyMatch(loc -> job.getLocation().toLowerCase().contains(loc.toLowerCase().trim()));
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
