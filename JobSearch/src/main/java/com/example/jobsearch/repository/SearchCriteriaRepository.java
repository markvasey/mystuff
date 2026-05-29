package com.example.jobsearch.repository;

import com.example.jobsearch.entity.SearchCriteria;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.UUID;

public interface SearchCriteriaRepository extends JpaRepository<SearchCriteria, UUID> {
    List<SearchCriteria> findByActiveTrue();
}
