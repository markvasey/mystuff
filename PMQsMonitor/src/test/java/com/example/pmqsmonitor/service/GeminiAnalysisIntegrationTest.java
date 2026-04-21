package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.Utterance;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
@ActiveProfiles("local") // Use 'local' to get your real API key
class GeminiAnalysisIntegrationTest {

    @Autowired
    private AnalysisService analysisService;

    @Test
    void testRealGeminiAnalysis() {
        // Arrange
        Utterance question = new Utterance();
        question.setSpeakerName("Kemi Badenoch");
        question.setText("That was a very interesting answer from the Prime Minister. Lord Robertson, who authored the Government’s strategic defence review, has said that the Prime Minister has a “corrosive complacency” when it comes to defence. Why did he say that?");

        Utterance answer = new Utterance();
        answer.setSpeakerName("Keir Starmer");
        answer.setText("""
                Let me start by saying that I respect Lord Robertson, and I thank him again for carrying out the strategic review. My responsibility is to keep the British people safe, and that is a duty I take seriously. That is why I do not agree with his comments.
                
                Last February—seven months after taking office—I took the decision to increase defence spending from 2.3% to 2.6%, which was paid for by a difficult decision on overseas aid. Last June at the NATO summit, I committed to raising core defence spending to 3.5%. Last November, the Budget committed record funding to defence. I reaffirm those commitments now.
                
                The strategic defence review is a 10-year blueprint for national security. The defence investment plan will put that into effect, and it will be published as soon as possible. We need to get it right. We inherited plans that were uncosted and undeliverable, and we are not going to repeat those mistakes.
                """);

        // Act
        System.out.println("Calling Gemini for real analysis...");
        analysisService.analyzeUtterance(question, answer);

        // Assert
        assertNotNull(answer.getAnalysisResult(), "Analysis result should be populated");
        System.out.println("ANALYSIS SUCCESSFUL!");

        assertEquals("Defensive",answer.getAnalysisResult().getSentiment());
        assertTrue(answer.getAnalysisResult().getCompleteness()>0);
        assertFalse(answer.getAnalysisResult().getRational().isEmpty());

        System.out.println("Sentiment: " + answer.getAnalysisResult().getSentiment());
        System.out.println("Completeness: " + answer.getAnalysisResult().getCompleteness() + "%");
        System.out.println("Reasoning: " + answer.getAnalysisResult().getRational());
    }
}
