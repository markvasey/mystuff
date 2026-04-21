package com.example.pmqsmonitor.repository;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import org.springframework.test.context.ActiveProfiles;

import java.time.LocalDateTime;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@DataJpaTest
@ActiveProfiles("test")
class AnalysisResultRepositoryTest {

    @Autowired
    private AnalysisResultRepository analysisResultRepository;

    @Autowired
    private UtteranceRepository utteranceRepository;

    @Test
    void testSaveAnalysisResult() {
        // 1. Create and save an Utterance first (FK dependency)
        Utterance utterance = new Utterance();
        utterance.setExternalId("test-gid-999");
        utterance.setText("Sample question text");
        utterance.setSpeakerName("Test MP");
        utterance = utteranceRepository.save(utterance);
        assertNotNull(utterance.getId());

        // 2. Create and save an AnalysisResult linked to that Utterance
        AnalysisResult result = new AnalysisResult();
        result.setUtterance(utterance);
        result.setSentiment("Positive");
        result.setTone("Formal");
        result.setCompleteness(85);
        result.setRelevance(90);
        result.setDirectAnswer(true);
        result.setDiversionTactics(List.of("None"));
        result.setPointsAnswered(List.of("Point A", "Point B"));
        result.setPointsMissed(List.of("Point C"));
        result.setRational("This is a solid answer.");
        result.setAnalyzedAt(LocalDateTime.now());

        // 3. Attempt save
        AnalysisResult savedResult = analysisResultRepository.save(result);
        
        // 4. Flush to force database interaction and catch mapping errors
        analysisResultRepository.flush();

        // 5. Assertions
        assertNotNull(savedResult.getId(), "ID should be generated");
        assertEquals("Positive", savedResult.getSentiment());
        assertEquals(utterance.getId(), savedResult.getUtterance().getId(), "Relationship should be preserved");
        
        // Verify retrieval
        AnalysisResult fetched = analysisResultRepository.findById(savedResult.getId()).orElse(null);
        assertNotNull(fetched);
        assertEquals(2, fetched.getPointsAnswered().size());
        assertEquals("Point A", fetched.getPointsAnswered().get(0));
    }
}
