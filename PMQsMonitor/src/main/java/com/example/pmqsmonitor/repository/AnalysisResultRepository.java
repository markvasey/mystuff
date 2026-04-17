package com.example.pmqsmonitor.repository;

import com.example.pmqsmonitor.model.AnalysisResult;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface AnalysisResultRepository extends JpaRepository<AnalysisResult, String> {
}
