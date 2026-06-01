package com.example.jobsearch.controller;

import com.example.jobsearch.entity.Person;
import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.PersonRepository;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import java.util.UUID;

@Controller
public class SearchCriteriaController {

    private final SearchCriteriaRepository criteriaRepository;
    private final PersonRepository personRepository;

    public SearchCriteriaController(SearchCriteriaRepository criteriaRepository, PersonRepository personRepository) {
        this.criteriaRepository = criteriaRepository;
        this.personRepository = personRepository;
    }

    @GetMapping("/criteria")
    public String listCriteria(Model model) {
        model.addAttribute("criteriaList", criteriaRepository.findAll());
        model.addAttribute("people", personRepository.findAll().stream()
                .sorted(java.util.Comparator.comparing(com.example.jobsearch.entity.Person::getName))
                .toList());
        return "criteria";
    }

    @PostMapping("/criteria/add")
    public String addCriteria(@RequestParam UUID personId,
                               @RequestParam String town, 
                               @RequestParam String keywords,
                               @RequestParam(required = false) String category,
                               @RequestParam(defaultValue = "false") boolean partTime,
                               @RequestParam(defaultValue = "5") int radius) {
        personRepository.findById(personId).ifPresent(person -> {
            SearchCriteria criteria = new SearchCriteria();
            criteria.setPerson(person);
            criteria.setTown(town);
            criteria.setKeywords(keywords);
            criteria.setCategory(category);
            criteria.setPartTime(partTime);
            criteria.setRadius(radius);
            criteriaRepository.save(criteria);
        });
        return "redirect:/criteria";
    }

    @PostMapping("/criteria/update")
    public String updateCriteria(@RequestParam java.util.UUID id,
                                 @RequestParam String keywords,
                                 @RequestParam String category,
                                 @RequestParam int radius,
                                 @RequestParam boolean partTime,
                                 @RequestParam boolean active) {
        criteriaRepository.findById(id).ifPresent(c -> {
            c.setKeywords(keywords);
            c.setCategory(category);
            c.setRadius(radius);
            c.setPartTime(partTime);
            c.setActive(active);
            criteriaRepository.save(c);
        });
        return "redirect:/criteria";
    }

    @PostMapping("/criteria/delete")
    public String deleteCriteria(@RequestParam java.util.UUID id) {
        criteriaRepository.deleteById(id);
        return "redirect:/criteria";
    }

    @PostMapping("/person/update-settings")
    public String updatePersonSettings(@RequestParam java.util.UUID personId,
                                       @RequestParam String blacklist,
                                       @RequestParam String locationBlacklist,
                                       @RequestParam String possibleList) {
        personRepository.findById(personId).ifPresent(p -> {
            p.setBlacklist(blacklist);
            p.setLocationBlacklist(locationBlacklist);
            p.setPossibleList(possibleList);
            personRepository.save(p);
        });
        return "redirect:/criteria";
    }
}
