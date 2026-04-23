package com.markvasey.mysearch.service;

import com.markvasey.mysearch.model.SearchItem;
import com.markvasey.mysearch.repository.SearchItemRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Service
public class SearchService {

    private final SearchItemRepository searchItemRepository;

    public SearchService(SearchItemRepository searchItemRepository) {
        this.searchItemRepository = searchItemRepository;
    }

    public List<SearchItem> search(String query) {
        if (query == null || query.trim().isEmpty()) {
            return List.of();
        }
        return searchItemRepository.search(query);
    }

    public long getTotalItems() {
        return searchItemRepository.count();
    }

    public String getDatabaseSize() {
        return searchItemRepository.getDatabaseSize();
    }

    public Optional<SearchItem> findById(UUID id) {
        return searchItemRepository.findById(id);
    }
}
