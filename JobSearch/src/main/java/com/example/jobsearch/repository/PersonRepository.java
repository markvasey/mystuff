package com.example.jobsearch.repository;

import com.example.jobsearch.entity.Person;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;
import java.util.UUID;

public interface PersonRepository extends JpaRepository<Person, UUID> {
    Optional<Person> findByName(String name);
}
