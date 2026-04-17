package com.example.pmqsmonitor.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "utterances")
public class Utterance {

    @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    @Column(columnDefinition = "VARCHAR(255)")
    private String id;

    @Column(name = "external_id", columnDefinition = "VARCHAR(255)")
    private String externalId;

    @Column(name = "speaker_name", columnDefinition = "VARCHAR(255)")
    private String speakerName;

    @Column(name = "speaker_id", columnDefinition = "VARCHAR(255)")
    private String speakerId;

    @Column(columnDefinition = "TEXT")
    private String text;

    @Column(name = "date_time", columnDefinition = "TIMESTAMP")
    private LocalDateTime dateTime;

    @Column(columnDefinition = "VARCHAR(50)")
    private String type; // e.g., "question", "answer"
    
    @Column(columnDefinition = "VARCHAR(50)")
    private String hdate;

    @Column(columnDefinition = "VARCHAR(50)")
    private String htime;

    @Column(columnDefinition = "TEXT")
    private String listurl;

    @Column(columnDefinition = "VARCHAR(100)")
    private String party;

    @Column(columnDefinition = "VARCHAR(50)")
    private String house;

    @Column(columnDefinition = "TEXT")
    private String office;

    @Column(name = "parent_body", columnDefinition = "TEXT")
    private String parentBody;

    @Column(name = "debate_type", columnDefinition = "VARCHAR(100)")
    private String debateType;

    @Column(name = "is_starmer", columnDefinition = "BOOLEAN")
    private boolean isStarmer;

    @Column(name = "is_representative", columnDefinition = "BOOLEAN")
    private boolean isRepresentative;

    @OneToOne(mappedBy = "utterance", cascade = CascadeType.ALL)
    private AnalysisResult analysisResult;

    // Getters and Setters
    public String getId() { return id; }
    public void setId(String id) { this.id = id; }
    public String getExternalId() { return externalId; }
    public void setExternalId(String externalId) { this.externalId = externalId; }
    public String getSpeakerName() { return speakerName; }
    public void setSpeakerName(String speakerName) { this.speakerName = speakerName; }
    public String getSpeakerId() { return speakerId; }
    public void setSpeakerId(String speakerId) { this.speakerId = speakerId; }
    public String getText() { return text; }
    public void setText(String text) { this.text = text; }
    public LocalDateTime getDateTime() { return dateTime; }
    public void setDateTime(LocalDateTime dateTime) { this.dateTime = dateTime; }
    public String getType() { return type; }
    public void setType(String type) { this.type = type; }
    public String getHdate() { return hdate; }
    public void setHdate(String hdate) { this.hdate = hdate; }
    public String getHtime() { return htime; }
    public void setHtime(String htime) { this.htime = htime; }
    public String getListurl() { return listurl; }
    public void setListurl(String listurl) { this.listurl = listurl; }
    public String getParty() { return party; }
    public void setParty(String party) { this.party = party; }
    public String getHouse() { return house; }
    public void setHouse(String house) { this.house = house; }
    public String getOffice() { return office; }
    public void setOffice(String office) { this.office = office; }
    public String getParentBody() { return parentBody; }
    public void setParentBody(String parentBody) { this.parentBody = parentBody; }
    public String getDebateType() { return debateType; }
    public void setDebateType(String debateType) { this.debateType = debateType; }
    public boolean isStarmer() { return isStarmer; }
    public void setStarmer(boolean starmer) { isStarmer = starmer; }
    public boolean isRepresentative() { return isRepresentative; }
    public void setRepresentative(boolean representative) { isRepresentative = representative; }
    public AnalysisResult getAnalysisResult() { return analysisResult; }
    public void setAnalysisResult(AnalysisResult analysisResult) { this.analysisResult = analysisResult; }
}
