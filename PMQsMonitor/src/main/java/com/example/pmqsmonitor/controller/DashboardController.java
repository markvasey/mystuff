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
    public String dashboard(Model model) {
        model.addAttribute("utterances", pmqsService.getLatestPMQs());
        return "dashboard";
    }


    @PostMapping("/scrape")
    public String scrape() {
        historicalScraperService.scrape2026();
        return "redirect:/";
    }
}
