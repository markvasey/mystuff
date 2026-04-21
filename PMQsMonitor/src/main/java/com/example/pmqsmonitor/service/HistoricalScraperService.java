package com.example.pmqsmonitor.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Service
public class HistoricalScraperService {

    private static final Logger log = LoggerFactory.getLogger(HistoricalScraperService.class);
    private final TWFYClient twfyClient;
    private final PMQsService pmqsService;

    public HistoricalScraperService(TWFYClient twfyClient, PMQsService pmqsService) {
        this.twfyClient = twfyClient;
        this.pmqsService = pmqsService;
    }

    public List<String> identify2026WednesdayFiles() {
        log.info("Identifying Wednesday XML files for 2026...");
        java.util.Set<String> identifiedFiles = new java.util.HashSet<>();
        
        // 1. Get the directory index
        String index = twfyClient.getXmlDirectoryIndex().block();
        if (index == null) return new ArrayList<>();

        // 2. Identify all Wednesday dates in 2026
        List<String> wednesdays = new ArrayList<>();
        LocalDate current = LocalDate.of(2026, 1, 1);
        LocalDate today = LocalDate.now();
        while (current.isBefore(today)) {
            if (current.getDayOfWeek() == DayOfWeek.WEDNESDAY) {
                wednesdays.add(current.format(DateTimeFormatter.ISO_LOCAL_DATE));
            }
            current = current.plusDays(1);
        }

        // 3. Find XML files that match these Wednesdays
        for (String wednesday : wednesdays) {
            Pattern filePattern = Pattern.compile("debates" + wednesday + "[a-z]\\.xml");
            Matcher fileMatcher = filePattern.matcher(index);
            
            while (fileMatcher.find()) {
                String filename = fileMatcher.group();
                identifiedFiles.add(filename);
            }
        }
        return new ArrayList<>(identifiedFiles);
    }

    public void scrape2026() {
        List<String> files = identify2026WednesdayFiles();
        log.info("Starting historical scrape for {} identified files...", files.size());
        
        for (String filename : files) {
            //log.info("Processing XML: {}", filename);
            processXmlFile(filename);
        }
    }

    public List<String> processXmlFile(String filename) {
        String xml = twfyClient.getRawXmlFile(filename).block();
        return parseGidsFromXml(xml);
    }

    private boolean isTestMode = false;

    public void setTestMode(boolean testMode) {
        this.isTestMode = testMode;
    }

    public List<String> parseGidsFromXml(String xml) {
        List<String> gids = new ArrayList<>();
        if (xml == null) return gids;

        // Find the block: Prime Minister heading -> Speech -> Engagements heading
        Pattern pmqPattern = Pattern.compile(
            "<major-heading[^>]*>\\s*Prime Minister\\s*</major-heading>\\s*<speech[^>]*id=\"([^\"]*)\"", 
            Pattern.DOTALL
        );
        Matcher m = pmqPattern.matcher(xml);

        while (m.find()) {
            String speechId = m.group(1);
            String apiGid = speechId.replace("uk.org.publicwhip/debate/", "");
            gids.add(apiGid);
            
            log.info("Found PMQs Speech GID: {}", apiGid);
            
            if (!isTestMode) {
                log.info("Fetching transcript for API GID: {}", apiGid);
                twfyClient.getFullDebateByGid(apiGid)
                        .doOnNext(rows -> {
                            if (rows != null && !rows.isEmpty()) {
                                log.info("HistoricalScraperService.parseGidsFromXml returned " + rows.size() + " rows");
                                pmqsService.processRows(rows);
                            }
                        })
                        .block();
            }
        }
        return gids;
    }
}

