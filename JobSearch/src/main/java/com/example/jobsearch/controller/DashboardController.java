package com.example.jobsearch.controller;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.JobListingRepository;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import com.example.jobsearch.service.JobPollerService;
import com.example.jobsearch.service.EmailService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Controller
public class DashboardController {

    private final JobListingRepository jobListingRepository;
    private final SearchCriteriaRepository criteriaRepository;
    private final JobPollerService jobPollerService;
    private final EmailService emailService;

    public DashboardController(JobListingRepository jobListingRepository,
                               SearchCriteriaRepository criteriaRepository,
                               JobPollerService jobPollerService,
                               EmailService emailService) {
        this.jobListingRepository = jobListingRepository;
        this.criteriaRepository = criteriaRepository;
        this.jobPollerService = jobPollerService;
        this.emailService = emailService;
    }

    @GetMapping("/")
    public String dashboard(@RequestParam(defaultValue = "40") int minScore, 
                            @RequestParam(defaultValue = "ACTIVE") String status,
                            @RequestParam(required = false) String town,
                            Model model) {
        
        // Get all unique towns from active criteria for the tabs
        List<String> activeTowns = criteriaRepository.findByActiveTrue().stream()
                .map(SearchCriteria::getTown)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        // Default to the first town if none specified and status is ACTIVE
        if ("ACTIVE".equals(status) && town == null && !activeTowns.isEmpty()) {
            town = activeTowns.get(0);
        }

        final String finalTown = town;
        List<JobListing> allJobsInTab = jobListingRepository.findAll().stream()
                .filter(j -> status.equals(j.getStatus()))
                .filter(j -> {
                    if ("ACTIVE".equals(status) && finalTown != null) {
                        return finalTown.equalsIgnoreCase(j.getTown());
                    }
                    return true;
                })
                .collect(Collectors.toList());

        List<JobListing> filteredJobs = allJobsInTab.stream()
                .filter(j -> j.getRelevanceScore() == null || j.getRelevanceScore() >= minScore)
                .collect(Collectors.toList()); 
        
        filteredJobs.sort((a, b) -> {
            Integer scoreA = a.getRelevanceScore() != null ? a.getRelevanceScore() : 0;
            Integer scoreB = b.getRelevanceScore() != null ? b.getRelevanceScore() : 0;
            return scoreB.compareTo(scoreA); // Descending
        });
        
        model.addAttribute("jobs", filteredJobs);
        model.addAttribute("totalInTab", allJobsInTab.size());
        model.addAttribute("filteredCount", filteredJobs.size());
        model.addAttribute("minScore", minScore);
        model.addAttribute("currentStatus", status);
        model.addAttribute("currentTown", town);
        model.addAttribute("activeTowns", activeTowns);
        model.addAttribute("criteriaCount", criteriaRepository.count());
        return "dashboard";
    }

    @PostMapping("/jobs/archive")
    public String archiveJob(@RequestParam UUID id, 
                             @RequestParam(required = false) String town,
                             @RequestParam(defaultValue = "40") int minScore) {
        jobListingRepository.findById(id).ifPresent(job -> {
            job.setStatus("ARCHIVED");
            jobListingRepository.save(job);
        });
        return "redirect:/?status=ACTIVE&minScore=" + minScore + (town != null ? "&town=" + town : "");
    }

    @PostMapping("/jobs/email")
    public String emailJob(@RequestParam UUID id, 
                           @RequestParam(required = false) String town,
                           @RequestParam(defaultValue = "40") int minScore) {
        jobListingRepository.findById(id).ifPresent(job -> {
            emailService.sendJobNotification(job);
            job.setStatus("EMAILED");
            jobListingRepository.save(job);
        });
        return "redirect:/?status=ACTIVE&minScore=" + minScore + (town != null ? "&town=" + town : "");
    }

    @GetMapping("/poll")
    public String triggerPoll(@RequestParam(required = false) String town,
                              @RequestParam(defaultValue = "40") int minScore) {
        jobPollerService.pollJobs();
        return "redirect:/?minScore=" + minScore + (town != null ? "&town=" + town : "");
    }
}
