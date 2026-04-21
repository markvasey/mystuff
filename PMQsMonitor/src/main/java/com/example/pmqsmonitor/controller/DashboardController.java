package com.example.pmqsmonitor.controller;

import com.example.pmqsmonitor.service.PMQsService;
import com.example.pmqsmonitor.service.HistoricalScraperService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class DashboardController {

    private final PMQsService pmqsService;
    private final HistoricalScraperService historicalScraperService;

    public DashboardController(PMQsService pmqsService, HistoricalScraperService historicalScraperService) {
        this.pmqsService = pmqsService;
        this.historicalScraperService = historicalScraperService;
    }

    @GetMapping("/")
    public String dashboard(@org.springframework.web.bind.annotation.RequestParam(required = false) String date, 
                            @org.springframework.web.bind.annotation.RequestParam(required = false) Boolean hideSpeaker,
                            @org.springframework.web.bind.annotation.RequestParam(required = false) String filterApplied,
                            Model model) {
        java.util.List<java.time.LocalDate> availableDates = pmqsService.getAvailableDates();
        model.addAttribute("availableDates", availableDates);

        // Logic: Default to true on first load. On form submission (filterApplied=true), 
        // if hideSpeaker is null, it means it was unchecked.
        boolean actualHideSpeaker = true;
        if ("true".equals(filterApplied)) {
            actualHideSpeaker = (hideSpeaker != null && hideSpeaker);
        }
        model.addAttribute("hideSpeaker", actualHideSpeaker);

        java.time.LocalDate selectedDate;
        if (date != null && !date.isEmpty()) {
            selectedDate = java.time.LocalDate.parse(date);
        } else if (!availableDates.isEmpty()) {
            selectedDate = availableDates.get(0);
        } else {
            selectedDate = null;
        }

        model.addAttribute("selectedDate", selectedDate);
        if (selectedDate != null) {
            model.addAttribute("utterances", pmqsService.getUtterancesByDate(selectedDate, actualHideSpeaker));
        } else {
            model.addAttribute("utterances", java.util.List.of());
        }
        return "dashboard";
    }


    @PostMapping("/scrape")
    public String scrape() {
        historicalScraperService.scrape2026();
        return "redirect:/";
    }
}
