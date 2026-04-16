package com.example.housepricemonitor.controller;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import com.example.housepricemonitor.model.PropertyTransaction;
import com.example.housepricemonitor.service.ComparisonConfigService;
import com.example.housepricemonitor.service.HousePriceAnalyticsService;
import com.example.housepricemonitor.service.HousePricePoller;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Controller
public class DashboardController {

    private final HousePriceAnalyticsService analyticsService;
    private final HousePricePoller poller;
    private final ComparisonConfigService comparisonConfigService;

    public DashboardController(HousePriceAnalyticsService analyticsService, 
                               HousePricePoller poller,
                               ComparisonConfigService comparisonConfigService) {
        this.analyticsService = analyticsService;
        this.poller = poller;
        this.comparisonConfigService = comparisonConfigService;
    }

    @GetMapping("/")
    public String index() {
        return "redirect:/dashboard";
    }

    @GetMapping("/dashboard")
    public String dashboard(Model model) {
        Map<String, List<PropertyTransaction>> transactionsByDistrict = analyticsService.getTransactionsByDistrict();
        
        Map<String, Object> stats = new HashMap<>();
        Map<String, Object> filteredStats = new HashMap<>();
        Map<String, List<PropertyTransaction>> filteredTransactions = new HashMap<>();

        for (String district : transactionsByDistrict.keySet()) {
            List<PropertyTransaction> allTxs = transactionsByDistrict.get(district);
            ComparatorsConfig.DistrictCriteria criteria = comparisonConfigService.getCriteriaForDistrict(district);
            
            // Sort allTxs so that similar ones are at the top, then by price descending
            allTxs.sort((a, b) -> {
                boolean aSimilar = analyticsService.isSimilar(a, criteria);
                boolean bSimilar = analyticsService.isSimilar(b, criteria);
                if (aSimilar && !bSimilar) return -1;
                if (!aSimilar && bSimilar) return 1;
                // Sub-sort by price descending
                return b.getPrice().compareTo(a.getPrice());
            });
            
            // Standard Stats
            Map<String, Object> districtStats = new HashMap<>();
            districtStats.put("count", allTxs.size());
            districtStats.put("avgPrice", analyticsService.calculateAveragePrice(allTxs));
            districtStats.put("avgPricePerSqm", analyticsService.calculateAveragePricePerSqm(allTxs));
            districtStats.put("avgPricePerRoom", analyticsService.calculateAveragePricePerRoom(allTxs));
            stats.put(district, districtStats);

            // Filtered (Similar) Stats
            if (criteria != null) {
                List<PropertyTransaction> similarTxs = analyticsService.filterByCriteria(allTxs, criteria);
                Map<String, Object> fStats = new HashMap<>();
                fStats.put("count", similarTxs.size());
                fStats.put("avgPrice", analyticsService.calculateAveragePrice(similarTxs));
                fStats.put("avgPricePerSqm", analyticsService.calculateAveragePricePerSqm(similarTxs));
                fStats.put("avgPricePerRoom", analyticsService.calculateAveragePricePerRoom(similarTxs));
                filteredStats.put(district, fStats);
                filteredTransactions.put(district, similarTxs);
            }
        }
        
        model.addAttribute("stats", stats);
        model.addAttribute("filteredStats", filteredStats);
        model.addAttribute("allTransactions", transactionsByDistrict);
        model.addAttribute("filteredTransactions", filteredTransactions);
        model.addAttribute("criteriaMap", comparisonConfigService.getAllCriteria());
        model.addAttribute("analyticsService", analyticsService); // To use isSimilar in Thymeleaf
        return "dashboard";
    }

    @GetMapping("/poll")
    public String poll() {
        poller.pollNewData();
        return "redirect:/dashboard";
    }
}
