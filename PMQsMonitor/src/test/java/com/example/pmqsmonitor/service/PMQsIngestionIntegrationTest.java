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
    void testFullIngestionFlowWithTestData() throws Exception {
        // Arrange
        String json = new String(java.nio.file.Files.readAllBytes(
                java.nio.file.Paths.get("src/main/resources/TestData/dataResponse.json")));
        
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper()
                .configure(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        TWFYClient.TWFYDebateResponse res = mapper.readValue(json, TWFYClient.TWFYDebateResponse.class);

        when(twfyClient.getPMQs()).thenReturn(Mono.just(res.rows));

        // Act
        pmqsService.pollNow();

        // Assert
        List<Utterance> all = utteranceRepository.findAll();
        assertFalse(all.isEmpty(), "Utterances should be saved to database");
        
        // Check for specific GID from the file (first one is 2011-12-22.12.0)
        Utterance first = utteranceRepository.findByExternalId("2011-12-22.12.0").orElse(null);
        assertNotNull(first, "First test record should be present");
        assertEquals("2011-12-22", first.getHdate());
    }
}
