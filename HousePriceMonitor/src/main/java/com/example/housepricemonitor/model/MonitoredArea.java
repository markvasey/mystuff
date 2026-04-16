package com.example.housepricemonitor.model;

import jakarta.persistence.*;

@Entity
@Table(name = "monitored_area")
public class MonitoredArea {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String postcodeDistrict;

    private String name;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPostcodeDistrict() { return postcodeDistrict; }
    public void setPostcodeDistrict(String postcodeDistrict) { this.postcodeDistrict = postcodeDistrict; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
}
