package com.example.jobsearch.config;

import com.example.jobsearch.entity.Person;
import com.example.jobsearch.repository.PersonRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;
import java.util.UUID;

@Component
public class DataMigrationRunner implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(DataMigrationRunner.class);
    private final JdbcTemplate jdbcTemplate;
    private final PersonRepository personRepository;

    public DataMigrationRunner(JdbcTemplate jdbcTemplate, PersonRepository personRepository) {
        this.jdbcTemplate = jdbcTemplate;
        this.personRepository = personRepository;
    }

    @Override
    @Transactional
    public void run(ApplicationArguments args) {
        log.info("Starting data migration check...");

        Person maya = personRepository.findByName("Maya").orElseGet(() -> {
            Person p = new Person();
            p.setName("Maya");
            p.setResumePath("MayaResume.txt");
            p.setEmail("mayasvasey@icloud.com");
            p.setBlacklist("Software, Developer, Engineer, Programmer, DevOps, Data Scientist, Data Analyst, Technician, Financial, Technical, Bid, Forklift, Global Asset Manager, German, Construction, Logistics, Defence, Receptionist, Warehouse, Export Agent");
            p.setLocationBlacklist("Southampton, Basingstoke, Portsmouth, Bournemouth, Reading, Andover");
            p.setPossibleList("Teacher, Teaching Assistant");
            return personRepository.save(p);
        });

        Person emily = personRepository.findByName("Emily").orElseGet(() -> {
            Person p = new Person();
            p.setName("Emily");
            p.setResumePath("EmilyCV.txt");
            p.setEmail("emilysvasey@icloud.com");
            p.setBlacklist("Software, Developer, Engineer, Programmer, DevOps, Data Scientist, Data Analyst, Technician, Financial, Technical, Bid, Forklift, Global Asset Manager, German, Construction, Logistics, Defence, Warehouse, Export Agent");
            p.setLocationBlacklist("Southampton, Basingstoke, Portsmouth, Bournemouth, Reading, Andover");
            p.setPossibleList("Teacher, Teaching Assistant, Playworker");
            return personRepository.save(p);
        });

        // Ensure Maya's email and defaults are up to date
        boolean mayaUpdated = false;
        if (maya.getEmail() == null || "markdvasey@icloud.com".equals(maya.getEmail())) {
            maya.setEmail("mayasvasey@icloud.com");
            mayaUpdated = true;
        }
        if (maya.getBlacklist() == null) {
            maya.setBlacklist("Software, Developer, Engineer, Programmer, DevOps, Data Scientist, Data Analyst, Technician, Financial, Technical, Bid, Forklift, Global Asset Manager, German, Construction, Logistics, Defence, Receptionist, Warehouse, Export Agent");
            mayaUpdated = true;
        }
        if (mayaUpdated) {
            personRepository.save(maya);
        }

        // Ensure Emily's email and defaults are up to date
        boolean emilyUpdated = false;
        if (emily.getEmail() == null) {
            emily.setEmail("emilysvasey@icloud.com");
            emilyUpdated = true;
        }
        if (emily.getBlacklist() == null) {
            emily.setBlacklist("Software, Developer, Engineer, Programmer, DevOps, Data Scientist, Data Analyst, Technician, Financial, Technical, Bid, Forklift, Global Asset Manager, German, Construction, Logistics, Defence, Warehouse, Export Agent");
            emilyUpdated = true;
        }
        if (emilyUpdated) {
            personRepository.save(emily);
        }

        // Migrate SearchCriteria
        int criteriaUpdated = jdbcTemplate.update(
            "UPDATE search_criteria SET person_id = ? WHERE person_id IS NULL",
            maya.getId()
        );
        if (criteriaUpdated > 0) {
            log.info("Migrated {} search criteria to Maya.", criteriaUpdated);
        }

        // Migrate JobListings to JobMatches
        try {
            // Check if old columns exist
            List<Map<String, Object>> unmigratedJobs = jdbcTemplate.queryForList(
                "SELECT id, status, relevance_score, match_reason, town FROM job_listing WHERE status IS NOT NULL"
            );

            int matchCount = 0;
            for (Map<String, Object> job : unmigratedJobs) {
                UUID jobId = (UUID) job.get("id");
                
                // Check if match already exists
                int exists = jdbcTemplate.queryForObject(
                    "SELECT COUNT(*) FROM job_match WHERE person_id = ? AND job_listing_id = ?",
                    Integer.class, maya.getId(), jobId
                );

                if (exists == 0) {
                    jdbcTemplate.update(
                        "INSERT INTO job_match (id, person_id, job_listing_id, status, relevance_score, match_reason, town) " +
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        UUID.randomUUID(),
                        maya.getId(),
                        jobId,
                        job.get("status"),
                        job.get("relevance_score"),
                        job.get("match_reason"),
                        job.get("town")
                    );
                    matchCount++;
                }
            }
            if (matchCount > 0) {
                log.info("Migrated {} jobs to job_match for Maya.", matchCount);
            }
            
            // Note: We don't drop the columns here because it might be risky, 
            // but we null them out so they aren't processed again.
            if (matchCount > 0) {
                jdbcTemplate.update("UPDATE job_listing SET status = NULL");
            }
            
        } catch (Exception e) {
            log.info("Job listing migration skipped or columns already dropped: {}", e.getMessage());
        }

        log.info("Data migration check complete.");
    }
}
