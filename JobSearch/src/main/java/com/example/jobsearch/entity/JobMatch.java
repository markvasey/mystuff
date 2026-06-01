package com.example.jobsearch.entity;

import jakarta.persistence.*;
import java.util.UUID;

@Entity
@Table(name = "job_match", uniqueConstraints = {
    @UniqueConstraint(columnNames = {"person_id", "job_listing_id"})
})
public class JobMatch {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne(optional = false)
    @JoinColumn(name = "person_id")
    private Person person;

    @ManyToOne(optional = false)
    @JoinColumn(name = "job_listing_id")
    private JobListing jobListing;

    @ManyToOne
    @JoinColumn(name = "search_criteria_id")
    private SearchCriteria searchCriteria;

    private String status = "ACTIVE"; // ACTIVE, ARCHIVED, EMAILED, POSSIBLE
    private Integer relevanceScore;
    
    @Column(columnDefinition = "TEXT")
    private String matchReason;
    
    private String town;

    // Getters and Setters
    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public Person getPerson() { return person; }
    public void setPerson(Person person) { this.person = person; }
    public JobListing getJobListing() { return jobListing; }
    public void setJobListing(JobListing jobListing) { this.jobListing = jobListing; }
    public SearchCriteria getSearchCriteria() { return searchCriteria; }
    public void setSearchCriteria(SearchCriteria searchCriteria) { this.searchCriteria = searchCriteria; }
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    public Integer getRelevanceScore() { return relevanceScore; }
    public void setRelevanceScore(Integer relevanceScore) { this.relevanceScore = relevanceScore; }
    public String getMatchReason() { return matchReason; }
    public void setMatchReason(String matchReason) { this.matchReason = matchReason; }
    public String getTown() { return town; }
    public void setTown(String town) { this.town = town; }
}
