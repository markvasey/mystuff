package com.example.housepricemonitor.repository;

import com.example.housepricemonitor.model.PropertyDetail;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface PropertyDetailRepository extends JpaRepository<PropertyDetail, Long> {
    Optional<PropertyDetail> findByPostcodeAndAddress(String postcode, String address);
}
