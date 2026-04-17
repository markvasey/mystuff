package com.example.pmqsmonitor.repository;

import com.example.pmqsmonitor.model.Utterance;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UtteranceRepository extends JpaRepository<Utterance, String> {
    Optional<Utterance> findByExternalId(String externalId);
}
