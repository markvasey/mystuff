package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import jakarta.annotation.PostConstruct;
import jakarta.xml.bind.JAXBContext;
import jakarta.xml.bind.Unmarshaller;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;

@Service
public class ComparisonConfigService {

    private Map<String, ComparatorsConfig.DistrictCriteria> criteriaMap = new HashMap<>();

    @PostConstruct
    public void loadConfig() {
        try {
            JAXBContext context = JAXBContext.newInstance(ComparatorsConfig.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            ClassPathResource resource = new ClassPathResource("HouseComparators.xml");
            ComparatorsConfig config = (ComparatorsConfig) unmarshaller.unmarshal(resource.getInputStream());
            for (ComparatorsConfig.DistrictCriteria district : config.getDistricts()) {
                criteriaMap.put(district.getPostcode(), district);
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to load HouseComparators.xml", e);
        }
    }

    public ComparatorsConfig.DistrictCriteria getCriteriaForDistrict(String district) {
        return criteriaMap.get(district);
    }

    public Map<String, ComparatorsConfig.DistrictCriteria> getAllCriteria() {
        return criteriaMap;
    }
}
