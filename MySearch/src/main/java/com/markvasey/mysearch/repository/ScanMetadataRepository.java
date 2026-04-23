package com.markvasey.mysearch.repository;

import com.markvasey.mysearch.model.ScanMetadata;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ScanMetadataRepository extends JpaRepository<ScanMetadata, String> {
}
