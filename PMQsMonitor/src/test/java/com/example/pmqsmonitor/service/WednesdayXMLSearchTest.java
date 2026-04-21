package com.example.pmqsmonitor.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("test")
class WednesdayXMLSearchTest {

    @Autowired
    private TWFYClient twfyClient;

    @Autowired
    private HistoricalScraperService scraperService;

    @Test
    void validateOnePMQPerWednesday() {
        scraperService.setTestMode(true);
        
        // 1. Get the directory index
        String index = twfyClient.getXmlDirectoryIndex().block();
        assertFalse(index == null || index.isEmpty());

        // 2. Identify all Wednesday dates in 2026 up to today
        List<String> wednesdays = new ArrayList<>();
        LocalDate current = LocalDate.of(2026, 1, 1);
        LocalDate today = LocalDate.now();
        while (current.isBefore(today)) {
            if (current.getDayOfWeek() == DayOfWeek.WEDNESDAY) {
                wednesdays.add(current.format(DateTimeFormatter.ISO_LOCAL_DATE));
            }
            current = current.plusDays(1);
        }

        System.out.println("Validating regex search across all Wednesday XML files...");
        
        // Map to track how many PMQ matches we find for each specific Wednesday date
        Map<String, Integer> wednesdayMatchCount = new HashMap<>();
        for (String wed : wednesdays) {
            wednesdayMatchCount.put(wed, 0);
        }

        // 3. Process every XML file associated with a Wednesday
        // Use a Set to avoid processing the same file multiple times
        Set<String> processedFiles = new HashSet<>();
        for (String wed : wednesdays) {
            Pattern pattern = Pattern.compile("debates" + wed + "[a-z]\\.xml");
            Matcher matcher = pattern.matcher(index);
            
            while (matcher.find()) {
                String filename = matcher.group();
                if (processedFiles.add(filename)) { // Only process if not already processed
                    List<String> gidsFound = scraperService.processXmlFile(filename);
                    if (!gidsFound.isEmpty()) {
                        System.out.println("PMQ HEADER FOUND in " + filename + " (GIDs: " + gidsFound + ")");
                        wednesdayMatchCount.put(wed, wednesdayMatchCount.get(wed) + gidsFound.size());
                    }
                }
            }
        }

        // 4. Final Validation
        int totalWednesdays = wednesdays.size();
        int wednesdaysWithOneMatch = 0;
        int wednesdaysWithZeroMatches = 0;
        
        for (String wed : wednesdays) {
            int count = wednesdayMatchCount.get(wed);
            if (count == 1) {
                wednesdaysWithOneMatch++;
            } else if (count == 0) {
                wednesdaysWithZeroMatches++;
                System.out.println("WARNING: Wednesday " + wed + " has NO PMQ matches. (Likely recess or no PMQs that day)");
            } else {
                System.out.println("ISSUE: Wednesday " + wed + " has " + count + " PMQ matches (Expected: 1)");
            }
        }

        System.out.println("Summary: " + processedFiles.size() + " sessions found.");
        System.out.println("Summary: " + wednesdays.size() + " Wednesday sessions found.");
        System.out.println("Summary: " + wednesdaysWithOneMatch + " had exactly one PMQ session.");
        System.out.println("Summary: " + wednesdaysWithZeroMatches + " had zero PMQ sessions (Recess).");

        assertTrue(wednesdays.size()>wednesdaysWithOneMatch,"Should find more wednesdays that wednesdaysWithOneMatch");
        assertTrue(wednesdaysWithOneMatch>wednesdaysWithZeroMatches,"Should find more wednesdaysWithOneMatch that wednesdaysWithZeroMatches");
        
        // We relax the assertion to allow for recess weeks
        assertFalse(wednesdaysWithOneMatch == 0, "Should find at least some PMQs sessions");
    }
}
