package com.example.jobsearch.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.UUID;

@Entity
@Table(name = "search_criteria")
public class SearchCriteria {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne
    @JoinColumn(name = "person_id")
    private Person person;

    @Column(nullable = false)
    private String town;

    @Column(nullable = false)
    private String keywords;

    private boolean active = true;

    private boolean partTime = true;
    private int radius = 5;
    private String category;

    private LocalDateTime lastPolledAt;

    // Getters and Setters
    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public Person getPerson() { return person; }
    public void setPerson(Person person) { this.person = person; }
    public String getTown() { return town; }
    public void setTown(String town) { this.town = town; }
    public String getKeywords() { return keywords; }
    public void setKeywords(String keywords) { this.keywords = keywords; }
    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }
    public boolean isPartTime() { return partTime; }
    public void setPartTime(boolean partTime) { this.partTime = partTime; }
    public int getRadius() { return radius; }
    public void setRadius(int radius) { this.radius = radius; }
    public String getCategory() { return category; }
    public void setCategory(String category) { this.category = category; }
    public LocalDateTime getLastPolledAt() { return lastPolledAt; }
    public void setLastPolledAt(LocalDateTime lastPolledAt) { this.lastPolledAt = lastPolledAt; }
}
