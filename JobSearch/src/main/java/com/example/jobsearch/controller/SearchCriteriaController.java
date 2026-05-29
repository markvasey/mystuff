package com.example.jobsearch.controller;

import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class SearchCriteriaController {

    private final SearchCriteriaRepository criteriaRepository;

    public SearchCriteriaController(SearchCriteriaRepository criteriaRepository) {
        this.criteriaRepository = criteriaRepository;
    }

    @GetMapping("/criteria")
    public String listCriteria(Model model) {
        model.addAttribute("criteriaList", criteriaRepository.findAll());
        return "criteria";
    }

    @PostMapping("/criteria/add")
    public String addCriteria(@RequestParam String town, 
                               @RequestParam String keywords,
                               @RequestParam(required = false) String category,
                               @RequestParam(defaultValue = "false") boolean partTime,
                               @RequestParam(defaultValue = "5") int radius) {
        SearchCriteria criteria = new SearchCriteria();
        criteria.setTown(town);
        criteria.setKeywords(keywords);
        criteria.setCategory(category);
        criteria.setPartTime(partTime);
        criteria.setRadius(radius);
        criteriaRepository.save(criteria);
        return "redirect:/criteria";
    }

    @PostMapping("/criteria/update")
    public String updateCriteria(@RequestParam java.util.UUID id,
                                 @RequestParam String keywords,
                                 @RequestParam String category,
                                 @RequestParam boolean active) {
        criteriaRepository.findById(id).ifPresent(c -> {
            c.setKeywords(keywords);
            c.setCategory(category);
            c.setActive(active);
            criteriaRepository.save(c);
        });
        return "redirect:/criteria";
    }
}
