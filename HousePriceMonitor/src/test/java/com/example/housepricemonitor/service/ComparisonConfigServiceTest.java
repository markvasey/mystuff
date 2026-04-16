package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import org.junit.jupiter.api.Test;
import org.springframework.core.io.ClassPathResource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

public class ComparisonConfigServiceTest {

    @Test
    void testLoadConfig() {
        ComparisonConfigService service = new ComparisonConfigService();
        service.loadConfig();

        java.util.List<com.example.housepricemonitor.dto.ComparatorsConfig.DistrictCriteria> allCriteria = service.getAllCriteria();
        assertNotNull(allCriteria);
        assertFalse(allCriteria.isEmpty(), "Should load criteria from HouseComparators.xml");

        // Verify one of the default districts exists in the list
        boolean found = allCriteria.stream().anyMatch(c -> c.getPostcode().equals("TS27") && c.getName().equals("Hartlepool"));
        assertTrue(found, "Should find Hartlepool in the loaded list");
    }
}
