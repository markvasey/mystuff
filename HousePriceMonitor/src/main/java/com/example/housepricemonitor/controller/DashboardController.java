package com.example.housepricemonitor.controller;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import com.example.housepricemonitor.model.PropertyTransaction;
import com.example.housepricemonitor.service.ComparisonConfigService;
import com.example.housepricemonitor.service.HousePriceAnalyticsService;
import com.example.housepricemonitor.service.HousePricePoller;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

import java.util.*;

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
        List<ComparatorsConfig.DistrictCriteria> allCriteria = comparisonConfigService.getAllCriteria();
        
        List<Map<String, Object>> dashboardCards = new ArrayList<>();
        Map<String, List<PropertyTransaction>> tabbedTransactions = new LinkedHashMap<>();

        for (ComparatorsConfig.DistrictCriteria criteria : allCriteria) {
            String postcode = criteria.getPostcode();
            List<PropertyTransaction> districtTxs = transactionsByDistrict.getOrDefault(postcode, new ArrayList<>());
            
            // Calculate Stats for this specific card
            Map<String, Object> card = new HashMap<>();
            card.put("criteria", criteria);
            card.put("marketCount", districtTxs.size());
            card.put("marketAvgPrice", analyticsService.calculateAveragePrice(districtTxs));
            card.put("marketAvgPricePerSqm", analyticsService.calculateAveragePricePerSqm(districtTxs));

            List<PropertyTransaction> similarTxs = analyticsService.filterByCriteria(districtTxs, criteria);
            card.put("similarCount", similarTxs.size());
            card.put("similarAvgPrice", analyticsService.calculateAveragePrice(similarTxs));
            card.put("similarAvgPricePerSqm", analyticsService.calculateAveragePricePerSqm(similarTxs));
            card.put("similarAvgPricePerRoom", analyticsService.calculateAveragePricePerRoom(similarTxs));
            
            dashboardCards.add(card);

            // Prepare sorted transactions for this criteria's tab
            List<PropertyTransaction> sortedTxs = new ArrayList<>(districtTxs);
            sortedTxs.sort((a, b) -> {
                boolean aSimilar = analyticsService.isSimilar(a, criteria);
                boolean bSimilar = analyticsService.isSimilar(b, criteria);
                if (aSimilar && !bSimilar) return -1;
                if (!aSimilar && bSimilar) return 1;
                return b.getPrice().compareTo(a.getPrice());
            });
            
            // Use Name + Postcode as unique tab key
            tabbedTransactions.put(criteria.getName() + " (" + postcode + ")", sortedTxs);
        }
        
        model.addAttribute("dashboardCards", dashboardCards);
        model.addAttribute("tabbedTransactions", tabbedTransactions);
        model.addAttribute("analyticsService", analyticsService);
        return "dashboard";
    }

    @GetMapping("/poll")
    public String poll() {
        poller.pollNewData();
        return "redirect:/dashboard";
    }
}
