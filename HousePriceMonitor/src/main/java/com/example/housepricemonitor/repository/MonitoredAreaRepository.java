package com.example.housepricemonitor.repository;

import com.example.housepricemonitor.model.MonitoredArea;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface MonitoredAreaRepository extends JpaRepository<MonitoredArea, Long> {
    Optional<MonitoredArea> findByPostcodeDistrict(String postcodeDistrict);
}
