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
    public String dashboard(@org.springframework.web.bind.annotation.RequestParam(required = false) String date, Model model) {
        java.util.List<java.time.LocalDate> availableDates = pmqsService.getAvailableDates();
        model.addAttribute("availableDates", availableDates);

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
            model.addAttribute("utterances", pmqsService.getUtterancesByDate(selectedDate));
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
