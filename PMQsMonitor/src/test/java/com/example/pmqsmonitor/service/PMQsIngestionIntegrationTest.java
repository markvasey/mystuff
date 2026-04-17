package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.UtteranceRepository;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.test.context.ActiveProfiles;
import reactor.core.publisher.Mono;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

@SpringBootTest
@ActiveProfiles("test")
class PMQsIngestionIntegrationTest {

    @Autowired
    private PMQsService pmqsService;

    @Autowired
    private UtteranceRepository utteranceRepository;

    @MockBean
    private TWFYClient twfyClient;

    @MockBean
    private AnalysisService analysisService;

    @Test
    void testFullIngestionFlow() {
        // Arrange
        TWFYClient.TWFYRow row = new TWFYClient.TWFYRow();
        row.gid = "test-gid-123";
        row.hdate = "2026-04-15";
        row.htime = "12:05:00";
        row.body = "<p>Test PMQs body</p>";
        row.listurl = "/test-url";
        row.debateType = "Oral questions";
        row.title = "Prime Minister's Question Time";

        TWFYClient.TWFYRow.SpeakerInfo speaker = new TWFYClient.TWFYRow.SpeakerInfo();
        speaker.personId = "25353";
        speaker.name = "Keir Starmer";
        speaker.party = "Labour";
        speaker.house = "1";
        
        TWFYClient.TWFYRow.OfficeInfo office = new TWFYClient.TWFYRow.OfficeInfo();
        office.position = "The Prime Minister";
        speaker.office = List.of(office);
        
        row.speaker = speaker;

        when(twfyClient.getPMQs()).thenReturn(Mono.just(List.of(row)));

        // Act
        pmqsService.pollNow();

        // Assert
        Utterance saved = utteranceRepository.findByExternalId("test-gid-123").orElse(null);
        assertNotNull(saved, "Utterance should be saved to database");
        assertEquals("2026-04-15", saved.getHdate());
        assertEquals("12:05:00", saved.getHtime());
        assertEquals("Keir Starmer", saved.getSpeakerName());
        assertEquals("25353", saved.getSpeakerId());
        assertEquals("Labour", saved.getParty());
        assertEquals("1", saved.getHouse());
        assertTrue(saved.getOffice().contains("The Prime Minister"));
        assertEquals("Oral questions", saved.getDebateType());
        assertTrue(saved.getListurl().contains("/test-url"));
        assertTrue(saved.isStarmer());
    }
}
