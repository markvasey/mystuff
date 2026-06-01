package com.example.jobsearch.repository;

import com.example.jobsearch.entity.JobMatch;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface JobMatchRepository extends JpaRepository<JobMatch, UUID> {
    Optional<JobMatch> findByPersonIdAndJobListingId(UUID personId, UUID jobListingId);
    List<JobMatch> findByPersonId(UUID personId);
    List<JobMatch> findByPersonIdAndStatus(UUID personId, String status);
}
