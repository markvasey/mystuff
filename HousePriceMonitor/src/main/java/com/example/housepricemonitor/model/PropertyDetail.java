package com.example.housepricemonitor.model;

import jakarta.persistence.*;
import java.math.BigDecimal;

@Entity
@Table(name = "property_detail")
public class PropertyDetail {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String postcode;

    private String address;

    private BigDecimal totalFloorArea;

    private Integer habitableRooms;

    private String propertyAgeBand;

    private String builtForm;

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getPostcode() { return postcode; }
    public void setPostcode(String postcode) { this.postcode = postcode; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public BigDecimal getTotalFloorArea() { return totalFloorArea; }
    public void setTotalFloorArea(BigDecimal totalFloorArea) { this.totalFloorArea = totalFloorArea; }
    public Integer getHabitableRooms() { return habitableRooms; }
    public void setHabitableRooms(Integer habitableRooms) { this.habitableRooms = habitableRooms; }
    public String getPropertyAgeBand() { return propertyAgeBand; }
    public void setPropertyAgeBand(String propertyAgeBand) { this.propertyAgeBand = propertyAgeBand; }
    public String getBuiltForm() { return builtForm; }
    public void setBuiltForm(String builtForm) { this.builtForm = builtForm; }
}
