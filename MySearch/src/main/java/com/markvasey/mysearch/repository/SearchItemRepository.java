package com.markvasey.mysearch.repository;

import com.markvasey.mysearch.model.SearchItem;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.UUID;

@Repository
public interface SearchItemRepository extends JpaRepository<SearchItem, UUID> {

    // PostGres Websearch syntax for Google-like search, sorted by latest first
    @Query(value = "SELECT * FROM search_items WHERE search_vector @@ websearch_to_tsquery('english', :query) " +
                   "ORDER BY item_date DESC, ts_rank(search_vector, websearch_to_tsquery('english', :query)) DESC", 
           nativeQuery = true)
    List<SearchItem> search(@Param("query") String query);

    @Query(value = "SELECT pg_size_pretty(pg_database_size(current_database()))", nativeQuery = true)
    String getDatabaseSize();

    boolean existsByExternalKey(String externalKey);

    List<SearchItem> findByItemDateBetweenOrderByItemDateDesc(LocalDateTime start, LocalDateTime end);

    List<SearchItem> findBySourceOrderByItemDateDesc(String source);

    List<SearchItem> findBySourceAndItemDateBetweenOrderByItemDateDesc(String source, LocalDateTime start, LocalDateTime end);
}
