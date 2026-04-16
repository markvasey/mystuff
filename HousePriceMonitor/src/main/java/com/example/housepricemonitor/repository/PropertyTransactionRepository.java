package com.example.housepricemonitor.repository;

import com.example.housepricemonitor.model.PropertyTransaction;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;

public interface PropertyTransactionRepository extends JpaRepository<PropertyTransaction, Long> {
    Optional<PropertyTransaction> findByTransactionId(String transactionId);
}
