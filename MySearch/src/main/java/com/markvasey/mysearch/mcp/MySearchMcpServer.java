package com.markvasey.mysearch.mcp;

import com.markvasey.mysearch.model.SearchItem;
import com.markvasey.mysearch.repository.SearchItemRepository;
import org.springaicommunity.mcp.annotation.McpTool;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

@Service
public class MySearchMcpServer {

    private final SearchItemRepository repository;

    public MySearchMcpServer(SearchItemRepository repository) {
        this.repository = repository;
    }

    @McpTool(description = "Search personal data using natural language or keywords (PostgreSQL Full-Text Search)")
    public String searchByText(String query) {
        List<SearchItem> items = repository.search(query);
        return formatResults(items);
    }

    @McpTool(description = "Search personal data within a specific date range. Dates should be in ISO format (yyyy-MM-ddTHH:mm:ss)")
    public String searchByDateRange(String start, String end) {
        LocalDateTime startTime = LocalDateTime.parse(start);
        LocalDateTime endTime = LocalDateTime.parse(end);
        List<SearchItem> items = repository.findByItemDateBetweenOrderByItemDateDesc(startTime, endTime);
        return formatResults(items);
    }

    @McpTool(description = "Filter personal data by source (EVERNOTE, DROPBOX, YAHOO_MAIL)")
    public String listBySource(String source) {
        List<SearchItem> items = repository.findBySourceOrderByItemDateDesc(source.toUpperCase());
        return formatResults(items);
    }

    @McpTool(description = "Get the full content and details of a specific item by its UUID")
    public String getItemDetails(String id) {
        return repository.findById(UUID.fromString(id))
                .map(item -> String.format("Title: %s\nSource: %s\nDate: %s\nContent:\n%s",
                        item.getTitle(), item.getSource(), item.getItemDate(), item.getContent()))
                .orElse("Item not found.");
    }

    private String formatResults(List<SearchItem> items) {
        if (items.isEmpty()) {
            return "No results found.";
        }
        return items.stream()
                .limit(20) // Limit results for AI context efficiency
                .map(item -> String.format("[%s] ID: %s | Source: %s | Date: %s | Title: %s\nSnippet: %s",
                        item.getItemDate().toLocalDate(),
                        item.getId(),
                        item.getSource(),
                        item.getItemDate(),
                        item.getTitle(),
                        item.getSnippet()))
                .collect(Collectors.joining("\n---\n"));
    }
}
