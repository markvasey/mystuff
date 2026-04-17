package com.example.pmqsmonitor.controller;

import com.example.pmqsmonitor.service.PMQsService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;

@Controller
public class DashboardController {

    private final PMQsService pmqsService;

    public DashboardController(PMQsService pmqsService) {
        this.pmqsService = pmqsService;
    }

    @GetMapping("/")
    public String dashboard(Model model) {
        model.addAttribute("utterances", pmqsService.getLatestPMQs());
        return "dashboard";
    }

    @PostMapping("/poll")
    public String poll() {
        pmqsService.pollNow();
        return "redirect:/";
    }
}
