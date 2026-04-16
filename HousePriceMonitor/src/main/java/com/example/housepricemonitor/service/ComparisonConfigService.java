package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import jakarta.annotation.PostConstruct;
import jakarta.xml.bind.JAXBContext;
import jakarta.xml.bind.Unmarshaller;
import org.springframework.core.io.ClassPathResource;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class ComparisonConfigService {

    private List<ComparatorsConfig.DistrictCriteria> allCriteria = new ArrayList<>();

    @PostConstruct
    public void loadConfig() {
        try {
            JAXBContext context = JAXBContext.newInstance(ComparatorsConfig.class);
            Unmarshaller unmarshaller = context.createUnmarshaller();
            ClassPathResource resource = new ClassPathResource("HouseComparators.xml");
            ComparatorsConfig config = (ComparatorsConfig) unmarshaller.unmarshal(resource.getInputStream());
            allCriteria = config.getDistricts();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load HouseComparators.xml", e);
        }
    }

    public List<ComparatorsConfig.DistrictCriteria> getAllCriteria() {
        return allCriteria;
    }
}
