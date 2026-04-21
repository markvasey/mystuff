package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.AnalysisResultRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;

import static org.junit.jupiter.api.Assertions.*;
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
