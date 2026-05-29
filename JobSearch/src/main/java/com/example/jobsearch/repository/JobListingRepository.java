package com.example.jobsearch.repository;

import com.example.jobsearch.entity.JobListing;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public interface JobListingRepository extends JpaRepository<JobListing, UUID> {
    Optional<JobListing> findByExternalIdAndSource(String externalId, String source);
    List<JobListing> findByTitleAndCompanyAndTown(String title, String company, String town);
}
