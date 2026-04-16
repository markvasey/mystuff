package com.example.housepricemonitor.dto;

import jakarta.xml.bind.annotation.*;
import java.util.List;

@XmlRootElement(name = "comparators")
@XmlAccessorType(XmlAccessType.FIELD)
public class ComparatorsConfig {

    @XmlElement(name = "district")
    private List<DistrictCriteria> districts;

    public List<DistrictCriteria> getDistricts() { return districts; }
    public void setDistricts(List<DistrictCriteria> districts) { this.districts = districts; }

    @XmlAccessorType(XmlAccessType.FIELD)
    public static class DistrictCriteria {
        @XmlAttribute
        private String postcode;
        @XmlAttribute
        private Double actualArea;
        @XmlAttribute
        private Integer actualRooms;
        private String name;
        private Integer minRooms;
        private Integer maxRooms;
        private String propertyType;
        private String ageBand;
        private Double minArea;
        private Double maxArea;

        // Getters and Setters
        public String getPostcode() { return postcode; }
        public void setPostcode(String postcode) { this.postcode = postcode; }
        public Double getActualArea() { return actualArea; }
        public void setActualArea(Double actualArea) { this.actualArea = actualArea; }
        public Integer getActualRooms() { return actualRooms; }
        public void setActualRooms(Integer actualRooms) { this.actualRooms = actualRooms; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public Integer getMinRooms() { return minRooms; }
        public void setMinRooms(Integer minRooms) { this.minRooms = minRooms; }
        public Integer getMaxRooms() { return maxRooms; }
        public void setMaxRooms(Integer maxRooms) { this.maxRooms = maxRooms; }
        public String getPropertyType() { return propertyType; }
        public void setPropertyType(String propertyType) { this.propertyType = propertyType; }
        public String getAgeBand() { return ageBand; }
        public void setAgeBand(String ageBand) { this.ageBand = ageBand; }
        public Double getMinArea() { return minArea; }
        public void setMinArea(Double minArea) { this.minArea = minArea; }
        public Double getMaxArea() { return maxArea; }
        public void setMaxArea(Double maxArea) { this.maxArea = maxArea; }
    }
}
