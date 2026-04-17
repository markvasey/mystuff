package com.example.pmqsmonitor.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "analysis_results")
public class AnalysisResult {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private String id;

    @OneToOne
    @JoinColumn(name = "utterance_id")
    private Utterance utterance;

    private String sentiment;
    private String tone;
    private int completeness;
    private int relevance;
    private boolean isDirectAnswer;

    @ElementCollection
    private List<String> diversionTactics;

    @ElementCollection
    private List<String> pointsAnswered;

    @ElementCollection
    private List<String> pointsMissed;

    @Column(columnDefinition = "TEXT")
    private String rational;

    private LocalDateTime analyzedAt;

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public Utterance getUtterance() { return utterance; }
    public void setUtterance(Utterance utterance) { this.utterance = utterance; }
    public String getSentiment() { return sentiment; }
    public void setSentiment(String sentiment) { this.sentiment = sentiment; }
    public String getTone() { return tone; }
    public void setTone(String tone) { this.tone = tone; }
    public int getCompleteness() { return completeness; }
    public void setCompleteness(int completeness) { this.completeness = completeness; }
    public int getRelevance() { return relevance; }
    public void setRelevance(int relevance) { this.relevance = relevance; }
    public boolean isDirectAnswer() { return isDirectAnswer; }
    public void setDirectAnswer(boolean directAnswer) { isDirectAnswer = directAnswer; }
    public List<String> getDiversionTactics() { return diversionTactics; }
    public void setDiversionTactics(List<String> diversionTactics) { this.diversionTactics = diversionTactics; }
    public List<String> getPointsAnswered() { return pointsAnswered; }
    public void setPointsAnswered(List<String> pointsAnswered) { this.pointsAnswered = pointsAnswered; }
    public List<String> getPointsMissed() { return pointsMissed; }
    public void setPointsMissed(List<String> pointsMissed) { this.pointsMissed = pointsMissed; }
    public String getRational() { return rational; }
    public void setRational(String rational) { this.rational = rational; }
    public LocalDateTime getAnalyzedAt() { return analyzedAt; }
    public void setAnalyzedAt(LocalDateTime analyzedAt) { this.analyzedAt = analyzedAt; }
}
