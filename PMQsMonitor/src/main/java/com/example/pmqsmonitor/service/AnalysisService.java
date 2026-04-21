package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.AnalysisResultRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;

@Service
public class AnalysisService {

    private static final Logger log = LoggerFactory.getLogger(AnalysisService.class);
    private final ChatClient chatClient;
    private final AnalysisResultRepository analysisResultRepository;

    public AnalysisService(ChatClient.Builder chatClientBuilder, 
                           AnalysisResultRepository analysisResultRepository) {
        this.chatClient = chatClientBuilder.build();
        this.analysisResultRepository = analysisResultRepository;
    }

    public void analyzeUtterance(Utterance question, Utterance answer) {
        if (answer.getAnalysisResult() != null) {
            return;
        }

        String qName = question.getSpeakerName() != null ? question.getSpeakerName() : "Unknown MP";
        String qText = question.getText() != null ? question.getText() : "[No question text available]";
        String aName = answer.getSpeakerName() != null ? answer.getSpeakerName() : "The Respondent";
        String aText = answer.getText() != null ? answer.getText() : "[No answer text available]";

        String prompt = String.format("""
                You are an expert political analyst. Analyze the following Prime Minister's Questions (PMQs) exchange.
                Focus specifically on the completeness and relevance of the answer given by %s.
                
                Question from %s: %s
                Answer from %s: %s
                
                Provide your analysis in JSON format with the following fields:
                - sentiment: String (e.g., "Defensive", "Honest", "Combative")
                - tone: String (brief description of the mood)
                - completeness: Integer (0-100)
                - relevance: Integer (0-100)
                - isDirectAnswer: Boolean
                - diversionTactics: List of Strings
                - pointsAnswered: List of Strings
                - pointsMissed: List of Strings
                - rational: String (detailed explanation for your scores)
                """, aName, qName, qText, aName, aText);

        try {
            log.info("Sending prompt to Gemini for answer: {}", answer.getExternalId());
            
            AnalysisResult result = chatClient.prompt()
                    .user(prompt)
                    .call()
                    .entity(new ParameterizedTypeReference<AnalysisResult>() {});

            if (result != null) {
                result.setUtterance(answer);
                result.setAnalyzedAt(LocalDateTime.now());
                analysisResultRepository.save(result);
                log.info("Analysis saved successfully for {}", answer.getExternalId());
            }
        } catch (Exception e) {
            log.error("Gemini Analysis Error: {}", e.getMessage(), e);
        }
    }
}
