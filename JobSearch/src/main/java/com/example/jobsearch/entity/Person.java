package com.example.jobsearch.entity;

import jakarta.persistence.*;
import java.util.UUID;

@Entity
@Table(name = "person")
public class Person {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @Column(nullable = false, unique = true)
    private String name;

    @Column(nullable = false)
    private String resumePath;

    @Column
    private String email;

    @Column(columnDefinition = "TEXT")
    private String blacklist;

    @Column(columnDefinition = "TEXT")
    private String locationBlacklist;

    @Column(columnDefinition = "TEXT")
    private String possibleList;

    // Getters and Setters
    public UUID getId() { return id; }
    public void setId(UUID id) { this.id = id; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public String getResumePath() { return resumePath; }
    public void setResumePath(String resumePath) { this.resumePath = resumePath; }
    public String getEmail() { return email; }
    public void setEmail(String email) { this.email = email; }
    public String getBlacklist() { return blacklist; }
    public void setBlacklist(String blacklist) { this.blacklist = blacklist; }
    public String getLocationBlacklist() { return locationBlacklist; }
    public void setLocationBlacklist(String locationBlacklist) { this.locationBlacklist = locationBlacklist; }
    public String getPossibleList() { return possibleList; }
    public void setPossibleList(String possibleList) { this.possibleList = possibleList; }
}
