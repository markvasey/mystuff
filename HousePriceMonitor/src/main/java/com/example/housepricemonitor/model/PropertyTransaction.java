package com.example.housepricemonitor.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDate;

@Entity
@Table(name = "property_transaction")
public class PropertyTransaction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(unique = true)
    private String transactionId;

    private BigDecimal price;

    private LocalDate transactionDate;

    private String postcode;

    private String address;

    private String propertyType;

    @ManyToOne
    @JoinColumn(name = "property_detail_id")
    private PropertyDetail propertyDetail;

    // Transient fields for the UI
    @Transient
    public BigDecimal getPricePerSqm() {
        if (propertyDetail != null && propertyDetail.getTotalFloorArea() != null && propertyDetail.getTotalFloorArea().compareTo(BigDecimal.ZERO) > 0) {
            return price.divide(propertyDetail.getTotalFloorArea(), 2, java.math.RoundingMode.HALF_UP);
        }
        return null;
    }

    @Transient
    public BigDecimal getPricePerRoom() {
        if (propertyDetail != null && propertyDetail.getHabitableRooms() != null && propertyDetail.getHabitableRooms() > 0) {
            return price.divide(BigDecimal.valueOf(propertyDetail.getHabitableRooms()), 2, java.math.RoundingMode.HALF_UP);
        }
        return null;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public String getTransactionId() { return transactionId; }
    public void setTransactionId(String transactionId) { this.transactionId = transactionId; }
    public BigDecimal getPrice() { return price; }
    public void setPrice(BigDecimal price) { this.price = price; }
    public LocalDate getTransactionDate() { return transactionDate; }
    public void setTransactionDate(LocalDate transactionDate) { this.transactionDate = transactionDate; }
    public String getPostcode() { return postcode; }
    public void setPostcode(String postcode) { this.postcode = postcode; }
    public String getAddress() { return address; }
    public void setAddress(String address) { this.address = address; }
    public String getPropertyType() { return propertyType; }
    public void setPropertyType(String propertyType) { this.propertyType = propertyType; }
    public PropertyDetail getPropertyDetail() { return propertyDetail; }
    public void setPropertyDetail(PropertyDetail propertyDetail) { this.propertyDetail = propertyDetail; }
}
