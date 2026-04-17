package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.UtteranceRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class PMQsServiceTest {

    private PMQsService pmqsService;
    private TWFYClient twfyClient;
    private UtteranceRepository utteranceRepository;
    private AnalysisService analysisService;

    @BeforeEach
    void setUp() {
        twfyClient = mock(TWFYClient.class);
        utteranceRepository = mock(UtteranceRepository.class);
        analysisService = mock(AnalysisService.class);
        pmqsService = new PMQsService(twfyClient, utteranceRepository, analysisService);
    }

    @Test
    void testPollNow_ProcessesAndAnalyzes() {
        // Arrange
        TWFYClient.TWFYRow questionRow = new TWFYClient.TWFYRow();
        questionRow.gid = "1";
        questionRow.hdate = "2024-03-20";
        questionRow.htime = "12:00:00";
        TWFYClient.TWFYRow.SpeakerInfo qSpeaker = new TWFYClient.TWFYRow.SpeakerInfo();
        qSpeaker.name = "MP Questioner";
        questionRow.speaker = qSpeaker;
        questionRow.body = "Is the NHS okay?";
        questionRow.debateType = "Oral questions";
        questionRow.title = "Prime Minister's Question Time";

        TWFYClient.TWFYRow answerRow = new TWFYClient.TWFYRow();
        answerRow.gid = "2";
        answerRow.hdate = "2024-03-20";
        answerRow.htime = "12:01:00";
        TWFYClient.TWFYRow.SpeakerInfo aSpeaker = new TWFYClient.TWFYRow.SpeakerInfo();
        aSpeaker.name = "Keir Starmer";
        aSpeaker.personId = "25353";
        answerRow.speaker = aSpeaker;
        answerRow.body = "Yes, we are investing.";
        answerRow.debateType = "Oral questions";
        answerRow.title = "Prime Minister's Question Time";

        TWFYClient.TWFYDebateResponse response = new TWFYClient.TWFYDebateResponse();
        response.rows = List.of(questionRow, answerRow);

        when(twfyClient.getPMQs()).thenReturn(Mono.just(response.rows));
        when(utteranceRepository.findByExternalId(anyString())).thenReturn(Optional.empty());
        when(utteranceRepository.save(any(Utterance.class))).thenAnswer(i -> i.getArguments()[0]);

        // Act
        pmqsService.pollNow();

        // Assert
        ArgumentCaptor<Utterance> utteranceCaptor = ArgumentCaptor.forClass(Utterance.class);
        verify(utteranceRepository, atLeast(2)).save(utteranceCaptor.capture());
        
        List<Utterance> saved = utteranceCaptor.getAllValues();
        // Since getLatestPMQs/pollNow might save multiple times due to linkQuestion logic, we check attributes
        Utterance q = saved.stream().filter(u -> "question".equals(u.getType())).findFirst().orElse(null);
        Utterance a = saved.stream().filter(u -> "answer".equals(u.getType())).findFirst().orElse(null);
        
        assertNotNull(q);
        assertNotNull(a);
        assertTrue(a.isStarmer());

        // Verify analysis was triggered for the answer
        verify(analysisService).analyzeUtterance(any(), eq(a));
    }
}
