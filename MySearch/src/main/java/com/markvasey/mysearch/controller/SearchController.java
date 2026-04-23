package com.markvasey.mysearch.controller;

import com.markvasey.mysearch.service.SearchService;
import com.markvasey.mysearch.service.SyncService;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
public class SearchController {

    private final SearchService searchService;
    private final SyncService syncService;

    public SearchController(SearchService searchService, SyncService syncService) {
        this.searchService = searchService;
        this.syncService = syncService;
    }

    @GetMapping("/")
    public String index(Model model, @RequestParam(value = "q", required = false) String query) {
        model.addAttribute("isSyncing", syncService.isSyncing());
        model.addAttribute("lastSyncTime", syncService.getLastSyncTimeFormatted());
        model.addAttribute("totalItems", searchService.getTotalItems());
        model.addAttribute("dbSize", searchService.getDatabaseSize());
        
        if (query != null && !query.trim().isEmpty()) {
            model.addAttribute("results", searchService.search(query));
            model.addAttribute("query", query);
            return "results";
        }
        return "index";
    }

    @GetMapping("/details/{id}")
    public String details(@PathVariable("id") java.util.UUID id, Model model) {
        searchService.findById(id).ifPresent(item -> model.addAttribute("item", item));
        return "details";
    }

    @PostMapping("/sync")
    public String triggerSync(RedirectAttributes redirectAttributes) {
        syncService.triggerSync();
        redirectAttributes.addFlashAttribute("message", "Sync triggered successfully!");
        return "redirect:/";
    }
}
