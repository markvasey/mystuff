package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import com.example.jobsearch.entity.Person;
import com.example.jobsearch.entity.SearchCriteria;
import com.example.jobsearch.repository.JobListingRepository;
import com.example.jobsearch.repository.JobMatchRepository;
import com.example.jobsearch.repository.PersonRepository;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
@Transactional
class MultiPersonIntegrationTest {

    @Autowired
    private PersonRepository personRepository;

    @Autowired
    private JobListingRepository jobListingRepository;

    @Autowired
    private JobMatchRepository jobMatchRepository;

    @Autowired
    private SearchCriteriaRepository criteriaRepository;

    private Person maya;
    private Person emily;

    @BeforeEach
    void setup() {
        jobMatchRepository.deleteAll();
        criteriaRepository.deleteAll();
        jobListingRepository.deleteAll();

        maya = personRepository.findByName("Maya").orElseGet(() -> {
            Person p = new Person();
            p.setName("Maya");
            p.setResumePath("MayaResume.txt");
            return personRepository.save(p);
        });

        emily = personRepository.findByName("Emily").orElseGet(() -> {
            Person p = new Person();
            p.setName("Emily");
            p.setResumePath("EmilyCV.txt");
            return personRepository.save(p);
        });
    }

    @Test
    void testJobsAreSharedButMatchesAreSeparate() {
        // Manually create one job listing
        JobListing job = new JobListing();
        job.setExternalId("test-123");
        job.setSource("Test");
        job.setTitle("Staff Member");
        job.setCompany("Test Co");
        job.setLocation("Winchester");
        job = jobListingRepository.save(job);

        // Create matches manually for isolation check
        JobMatch mayaMatch = new JobMatch();
        mayaMatch.setPerson(maya);
        mayaMatch.setJobListing(job);
        mayaMatch.setStatus("ACTIVE");
        mayaMatch.setTown("Winchester");
        jobMatchRepository.save(mayaMatch);

        JobMatch emilyMatch = new JobMatch();
        emilyMatch.setPerson(emily);
        emilyMatch.setJobListing(job);
        emilyMatch.setStatus("ACTIVE");
        emilyMatch.setTown("Winchester");
        jobMatchRepository.save(emilyMatch);

        // Verify 1 job, 2 matches
        assertEquals(1, jobListingRepository.count());
        assertEquals(2, jobMatchRepository.count());

        // Update Maya's status
        mayaMatch.setStatus("ARCHIVED");
        jobMatchRepository.save(mayaMatch);

        // Verify Emily's status is unchanged
        Optional<JobMatch> reloadedEmilyMatch = jobMatchRepository.findByPersonIdAndJobListingId(emily.getId(), job.getId());
        assertTrue(reloadedEmilyMatch.isPresent());
        assertEquals("ACTIVE", reloadedEmilyMatch.get().getStatus());
        
        Optional<JobMatch> reloadedMayaMatch = jobMatchRepository.findByPersonIdAndJobListingId(maya.getId(), job.getId());
        assertTrue(reloadedMayaMatch.isPresent());
        assertEquals("ARCHIVED", reloadedMayaMatch.get().getStatus());
    }
}
