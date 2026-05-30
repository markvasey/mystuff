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
import org.springframework.web.bind.annotation.ResponseBody;
import java.util.List;
import java.util.Map;
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
                            @RequestParam(required = false) String fragment,
                            Model model) {
        
        List<String> activeTowns = criteriaRepository.findByActiveTrue().stream()
                .map(SearchCriteria::getTown)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        if (("ACTIVE".equals(status) || "POSSIBLE".equals(status)) && town == null && !activeTowns.isEmpty()) {
            town = activeTowns.get(0);
        }

        final String finalTown = town;
        List<JobListing> allJobsInTab = jobListingRepository.findAll().stream()
                .filter(j -> status.equals(j.getStatus()))
                .filter(j -> {
                    if (("ACTIVE".equals(status) || "POSSIBLE".equals(status)) && finalTown != null) {
                        return finalTown.equalsIgnoreCase(j.getTown());
                    }
                    return true;
                })
                .collect(Collectors.toList());

        List<JobListing> filteredJobs = allJobsInTab.stream()
                .filter(j -> {
                    if ("ACTIVE".equals(status)) {
                        return j.getRelevanceScore() == null || j.getRelevanceScore() >= minScore;
                    }
                    return true; // No score filter for POSSIBLE, EMAILED, ARCHIVED
                })
                .collect(Collectors.toList()); 
        
        filteredJobs.sort((a, b) -> {
            Integer scoreA = a.getRelevanceScore() != null ? a.getRelevanceScore() : 0;
            Integer scoreB = b.getRelevanceScore() != null ? b.getRelevanceScore() : 0;
            return scoreB.compareTo(scoreA);
        });
        
        model.addAttribute("jobs", filteredJobs);
        model.addAttribute("totalInTab", allJobsInTab.size());
        model.addAttribute("filteredCount", filteredJobs.size());
        
        long totalActiveAllTowns = jobListingRepository.findAll().stream()
                .filter(j -> "ACTIVE".equals(j.getStatus()) || "POSSIBLE".equals(j.getStatus()))
                .count();
        model.addAttribute("totalActiveAllTowns", totalActiveAllTowns);

        model.addAttribute("currentMinScore", minScore);
        model.addAttribute("currentStatus", status);
        model.addAttribute("currentTown", town);
        model.addAttribute("activeTowns", activeTowns);
        model.addAttribute("criteriaCount", criteriaRepository.count());
        model.addAttribute("isPolling", jobPollerService.isPolling());

        if ("results".equals(fragment)) {
            return "dashboard :: resultsFragment";
        }
        return "dashboard";
    }

    @GetMapping("/api/status")
    @ResponseBody
    public Map<String, Object> getStatus() {
        long totalActive = jobListingRepository.findAll().stream()
                .filter(j -> "ACTIVE".equals(j.getStatus()) || "POSSIBLE".equals(j.getStatus()))
                .count();
        return Map.of(
            "totalActive", totalActive,
            "isPolling", jobPollerService.isPolling()
        );
    }

    @PostMapping("/jobs/archive")
    public String archiveJob(@RequestParam UUID id, 
                             @RequestParam(required = false) String town,
                             @RequestParam(defaultValue = "40", name = "minScore") int minScore) {
        jobListingRepository.findById(id).ifPresent(job -> {
            job.setStatus("ARCHIVED");
            jobListingRepository.save(job);
        });
        return "redirect:/?status=ACTIVE&minScore=" + minScore + (town != null ? "&town=" + town : "");
    }

    @PostMapping("/jobs/email")
    public String emailJob(@RequestParam UUID id, 
                           @RequestParam(required = false) String town,
                           @RequestParam(defaultValue = "40", name = "minScore") int minScore) {
        jobListingRepository.findById(id).ifPresent(job -> {
            emailService.sendJobNotification(job);
            job.setStatus("EMAILED");
            jobListingRepository.save(job);
        });
        return "redirect:/?status=ACTIVE&minScore=" + minScore + (town != null ? "&town=" + town : "");
    }

    @GetMapping("/poll")
    public String triggerPoll(@RequestParam(required = false) String town,
                              @RequestParam(defaultValue = "40", name = "minScore") int minScore) {
        jobPollerService.pollJobs();
        return "redirect:/?minScore=" + minScore + (town != null ? "?town=" + town : "");
    }
}
