package com.example.pmqsmonitor.repository;

import com.example.pmqsmonitor.model.Utterance;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface UtteranceRepository extends JpaRepository<Utterance, String> {
    Optional<Utterance> findByExternalId(String externalId);

    @org.springframework.data.jpa.repository.Query("SELECT DISTINCT CAST(u.dateTime AS date) FROM Utterance u WHERE u.hdate IS NOT NULL ORDER BY 1 DESC")
    List<java.sql.Date> findDistinctDates();
}
