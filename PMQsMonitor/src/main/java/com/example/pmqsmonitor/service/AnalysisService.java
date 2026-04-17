package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.AnalysisResultRepository;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Map;

@Service
public class AnalysisService {

    private final ChatClient chatClient;
    private final AnalysisResultRepository analysisResultRepository;

    public AnalysisService(ChatClient.Builder chatClientBuilder, AnalysisResultRepository analysisResultRepository) {
        this.chatClient = chatClientBuilder.build();
        this.analysisResultRepository = analysisResultRepository;
    }

    public void analyzeUtterance(Utterance question, Utterance answer) {
        if (answer.getAnalysisResult() != null) {
            return;
        }

        String prompt = """
                You are an expert political analyst. Analyze the following Prime Minister's Questions (PMQs) exchange.
                Focus specifically on the completeness and relevance of the answer given by Keir Starmer.
                
                Question from %s: %s
                Answer from %s: %s
                
                Provide your analysis in JSON format with the following fields:
                - sentiment: String (e.g., "Defensive", "Honest", "Combative")
                - tone: String (brief description of the mood)
                - completeness: Integer (0-100, how much of the question was actually addressed)
                - relevance: Integer (0-100, how relevant the answer was to the specific question)
                - isDirectAnswer: Boolean (true if they answered the question directly)
                - diversionTactics: List of Strings (any tactics like 'pivoting', 'blaming previous gov', 'changing subject')
                - pointsAnswered: List of Strings (specific points from the question that were answered)
                - pointsMissed: List of Strings (specific points from the question that were ignored)
                - rational: String (detailed explanation for your scores)
                """;

        String formattedPrompt = String.format(prompt, 
                question.getSpeakerName(), question.getText(), 
                answer.getSpeakerName(), answer.getText());

        AnalysisResult result = chatClient.prompt()
                .user(formattedPrompt)
                .call()
                .entity(new ParameterizedTypeReference<AnalysisResult>() {});

        if (result != null) {
            result.setUtterance(answer);
            result.setAnalyzedAt(LocalDateTime.now());
            analysisResultRepository.save(result);
        }
    }
}
