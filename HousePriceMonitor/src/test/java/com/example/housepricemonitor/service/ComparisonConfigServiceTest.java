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

        Map<String, ComparatorsConfig.DistrictCriteria> allCriteria = service.getAllCriteria();
        assertNotNull(allCriteria);
        assertFalse(allCriteria.isEmpty(), "Should load criteria from HouseComparators.xml");

        // Verify one of the default districts
        ComparatorsConfig.DistrictCriteria ts27 = service.getCriteriaForDistrict("TS27");
        assertNotNull(ts27);
        assertEquals("Hartlepool", ts27.getName());
    }
}
