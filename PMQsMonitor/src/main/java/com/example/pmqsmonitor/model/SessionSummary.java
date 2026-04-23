package com.example.pmqsmonitor.model;

import jakarta.persistence.*;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Table(name = "session_summaries")
public class SessionSummary {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    @Column(unique = true)
    private LocalDate sessionDate;

    @Column(columnDefinition = "TEXT")
    private String executiveSummary;

    private int totalAnalyzed;
    private double avgCompleteness;
    private double avgRelevance;
    private int directAnswers;

    private LocalDateTime calculatedAt;

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public LocalDate getSessionDate() { return sessionDate; }
    public void setSessionDate(LocalDate sessionDate) { this.sessionDate = sessionDate; }
    public String getExecutiveSummary() { return executiveSummary; }
    public void setExecutiveSummary(String executiveSummary) { this.executiveSummary = executiveSummary; }
    public int getTotalAnalyzed() { return totalAnalyzed; }
    public void setTotalAnalyzed(int totalAnalyzed) { this.totalAnalyzed = totalAnalyzed; }
    public double getAvgCompleteness() { return avgCompleteness; }
    public void setAvgCompleteness(double avgCompleteness) { this.avgCompleteness = avgCompleteness; }
    public double getAvgRelevance() { return avgRelevance; }
    public void setAvgRelevance(double avgRelevance) { this.avgRelevance = avgRelevance; }
    public int getDirectAnswers() { return directAnswers; }
    public void setDirectAnswers(int directAnswers) { this.directAnswers = directAnswers; }
    public LocalDateTime getCalculatedAt() { return calculatedAt; }
    public void setCalculatedAt(LocalDateTime calculatedAt) { this.calculatedAt = calculatedAt; }
}
