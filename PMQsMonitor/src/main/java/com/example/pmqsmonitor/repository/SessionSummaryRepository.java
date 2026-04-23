package com.example.pmqsmonitor.repository;

import com.example.pmqsmonitor.model.SessionSummary;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.Optional;

@Repository
public interface SessionSummaryRepository extends JpaRepository<SessionSummary, String> {
    Optional<SessionSummary> findBySessionDate(LocalDate sessionDate);
}
