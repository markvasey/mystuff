package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.AnalysisResultRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.ParameterizedTypeReference;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

class AnalysisServiceTest {

    private AnalysisService analysisService;
    private ChatClient chatClient;
    private AnalysisResultRepository analysisResultRepository;

    @BeforeEach
    void setUp() {
        chatClient = mock(ChatClient.class, RETURNS_DEEP_STUBS);
        ChatClient.Builder builder = mock(ChatClient.Builder.class);
        when(builder.build()).thenReturn(chatClient);
        
        analysisResultRepository = mock(AnalysisResultRepository.class);
        analysisService = new AnalysisService(builder, analysisResultRepository);
    }

    @Test
    void testAnalyzeUtterance_Success() {
        // Arrange
        Utterance question = new Utterance();
        question.setSpeakerName("John Doe");
        question.setText("Why is the sky blue?");

        Utterance answer = new Utterance();
        answer.setSpeakerName("Keir Starmer");
        answer.setText("Because of Rayleigh scattering.");

        AnalysisResult mockResult = new AnalysisResult();
        mockResult.setSentiment("Informative");
        mockResult.setCompleteness(100);
        mockResult.setRelevance(100);
        mockResult.setRational("Direct and scientifically accurate answer.");
        mockResult.setPointsAnswered(List.of("Sky color explanation"));

        // Mock the fluent API of ChatClient
        when(chatClient.prompt().user(anyString()).call().entity(any(ParameterizedTypeReference.class)))
                .thenReturn(mockResult);

        // Act
        analysisService.analyzeUtterance(question, answer);

        // Assert
        ArgumentCaptor<AnalysisResult> captor = ArgumentCaptor.forClass(AnalysisResult.class);
        verify(analysisResultRepository).save(captor.capture());
        
        AnalysisResult savedResult = captor.getValue();
        assertEquals("Informative", savedResult.getSentiment());
        assertEquals(100, savedResult.getCompleteness());
        assertEquals(answer, savedResult.getUtterance());
        assertNotNull(savedResult.getAnalyzedAt());
    }

    @Test
    void testAnalyzeUtterance_AlreadyAnalyzed_Skips() {
        // Arrange
        Utterance answer = new Utterance();
        answer.setAnalysisResult(new AnalysisResult());

        // Act
        analysisService.analyzeUtterance(new Utterance(), answer);

        // Assert
        verifyNoInteractions(chatClient);
        verifyNoInteractions(analysisResultRepository);
    }
}
