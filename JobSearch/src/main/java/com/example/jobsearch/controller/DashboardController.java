package com.example.jobsearch.controller;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import com.example.jobsearch.entity.Person;
import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.JobListingRepository;
import com.example.jobsearch.repository.JobMatchRepository;
import com.example.jobsearch.repository.PersonRepository;
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

    private final JobMatchRepository jobMatchRepository;
    private final PersonRepository personRepository;
    private final SearchCriteriaRepository criteriaRepository;
    private final JobPollerService jobPollerService;
    private final EmailService emailService;

    public DashboardController(JobMatchRepository jobMatchRepository,
                               PersonRepository personRepository,
                               SearchCriteriaRepository criteriaRepository,
                               JobPollerService jobPollerService,
                               EmailService emailService) {
        this.jobMatchRepository = jobMatchRepository;
        this.personRepository = personRepository;
        this.criteriaRepository = criteriaRepository;
        this.jobPollerService = jobPollerService;
        this.emailService = emailService;
    }

    @GetMapping("/")
    public String dashboard(@RequestParam(required = false) UUID personId,
                            @RequestParam(defaultValue = "40") int minScore, 
                            @RequestParam(defaultValue = "ACTIVE") String status,
                            @RequestParam(required = false) String town,
                            @RequestParam(required = false) String fragment,
                            Model model) {
        
        List<Person> people = personRepository.findAll();
        Person selectedPerson = null;
        
        if (personId != null) {
            selectedPerson = personRepository.findById(personId).orElse(null);
        }
        if (selectedPerson == null && !people.isEmpty()) {
            selectedPerson = people.get(0);
        }

        if (selectedPerson == null) {
            return "dashboard"; // Handle empty DB
        }

        Person finalSelectedPerson = selectedPerson;

        List<String> activeTowns = criteriaRepository.findByActiveTrue().stream()
                .filter(c -> finalSelectedPerson.equals(c.getPerson()))
                .map(SearchCriteria::getTown)
                .distinct()
                .sorted()
                .collect(Collectors.toList());

        if (("ACTIVE".equals(status) || "POSSIBLE".equals(status)) && town == null && !activeTowns.isEmpty()) {
            town = activeTowns.get(0);
        }

        final String finalTown = town;
        List<JobMatch> allMatchesForPerson = jobMatchRepository.findByPersonId(selectedPerson.getId());
        
        List<JobMatch> allJobsInTab = allMatchesForPerson.stream()
                .filter(m -> status.equals(m.getStatus()))
                .filter(m -> {
                    if (("ACTIVE".equals(status) || "POSSIBLE".equals(status)) && finalTown != null) {
                        return finalTown.equalsIgnoreCase(m.getTown());
                    }
                    return true;
                })
                .collect(Collectors.toList());

        List<JobMatch> filteredJobs = allJobsInTab.stream()
                .filter(m -> {
                    if ("ACTIVE".equals(status)) {
                        return m.getRelevanceScore() == null || m.getRelevanceScore() >= minScore;
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
        
        long totalActiveAllTowns = allMatchesForPerson.stream()
                .filter(m -> "ACTIVE".equals(m.getStatus()) || "POSSIBLE".equals(m.getStatus()))
                .count();
        model.addAttribute("totalActiveAllTowns", totalActiveAllTowns);

        model.addAttribute("people", people);
        model.addAttribute("selectedPerson", selectedPerson);
        model.addAttribute("selectedPersonId", selectedPerson.getId());
        model.addAttribute("currentMinScore", minScore);
        model.addAttribute("currentStatus", status);
        model.addAttribute("currentTown", town);
        model.addAttribute("activeTowns", activeTowns);
        
        long personCriteriaCount = criteriaRepository.findAll().stream().filter(c -> finalSelectedPerson.equals(c.getPerson())).count();
        model.addAttribute("criteriaCount", personCriteriaCount);
        model.addAttribute("isPolling", jobPollerService.isPolling());

        if ("results".equals(fragment)) {
            return "dashboard :: resultsFragment";
        }
        return "dashboard";
    }

    @GetMapping("/api/status")
    @ResponseBody
    public Map<String, Object> getStatus(@RequestParam(required = false) UUID personId) {
        long totalActive = 0;
        if (personId != null) {
            totalActive = jobMatchRepository.findByPersonId(personId).stream()
                .filter(m -> "ACTIVE".equals(m.getStatus()) || "POSSIBLE".equals(m.getStatus()))
                .count();
        } else {
             totalActive = jobMatchRepository.findAll().stream()
                .filter(m -> "ACTIVE".equals(m.getStatus()) || "POSSIBLE".equals(m.getStatus()))
                .count();
        }
        return Map.of(
            "totalActive", totalActive,
            "isPolling", jobPollerService.isPolling()
        );
    }

    @PostMapping("/jobs/archive")
    public String archiveJob(@RequestParam UUID id, 
                             @RequestParam UUID personId,
                             @RequestParam(required = false) String town,
                             @RequestParam(defaultValue = "40", name = "minScore") int minScore) {
        jobMatchRepository.findById(id).ifPresent(match -> {
            match.setStatus("ARCHIVED");
            jobMatchRepository.save(match);
        });
        return "redirect:/?personId=" + personId + "&status=ACTIVE&minScore=" + minScore + (town != null ? "&town=" + town : "");
    }

    @PostMapping("/jobs/archive-filtered")
    @ResponseBody
    public Map<String, Object> archiveFiltered(@RequestParam UUID personId,
                                              @RequestParam String town, 
                                              @RequestParam int minScore,
                                              @RequestParam(defaultValue = "ACTIVE") String status) {
        List<JobMatch> matchesToArchive = jobMatchRepository.findByPersonId(personId).stream()
                .filter(m -> status.equals(m.getStatus()))
                .filter(m -> town.equalsIgnoreCase(m.getTown()))
                .filter(m -> {
                    if ("ACTIVE".equals(status)) {
                        return m.getRelevanceScore() == null || m.getRelevanceScore() >= minScore;
                    }
                    return true;
                })
                .collect(Collectors.toList());
        
        int count = matchesToArchive.size();
        matchesToArchive.forEach(m -> m.setStatus("ARCHIVED"));
        jobMatchRepository.saveAll(matchesToArchive);
        
        return Map.of("archivedCount", count, "success", true);
    }

    @PostMapping("/jobs/email")
    public String emailJob(@RequestParam UUID id, 
                           @RequestParam UUID personId,
                           @RequestParam(required = false) String town,
                           @RequestParam(defaultValue = "40", name = "minScore") int minScore) {
        jobMatchRepository.findById(id).ifPresent(match -> {
            emailService.sendJobNotification(match); // Note: may want to update email service too
            match.setStatus("EMAILED");
            jobMatchRepository.save(match);
        });
        return "redirect:/?personId=" + personId + "&status=ACTIVE&minScore=" + minScore + (town != null ? "&town=" + town : "");
    }

    @GetMapping("/poll")
    public String triggerPoll(@RequestParam(required = false) UUID personId,
                              @RequestParam(required = false) String town,
                              @RequestParam(defaultValue = "40", name = "minScore") int minScore) {
        jobPollerService.pollJobs();
        return "redirect:/?" + (personId != null ? "personId=" + personId + "&" : "") + "minScore=" + minScore + (town != null ? "&town=" + town : "");
    }
}
