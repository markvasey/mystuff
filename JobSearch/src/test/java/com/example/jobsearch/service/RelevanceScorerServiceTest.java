package com.example.jobsearch.service;

import com.example.jobsearch.entity.JobListing;
import com.example.jobsearch.entity.JobMatch;
import com.example.jobsearch.entity.Person;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClient.ChatClientRequestSpec;
import org.springframework.ai.chat.client.ChatClient.CallResponseSpec;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class RelevanceScorerServiceTest {

    @Mock
    private ChatClient.Builder chatClientBuilder;

    @Mock
    private ChatClient chatClient;

    @Mock
    private ChatClientRequestSpec requestSpec;
    
    @Mock
    private CallResponseSpec responseSpec;

    private RelevanceScorerService relevanceScorerService;

    @BeforeEach
    void setUp() {
        when(chatClientBuilder.build()).thenReturn(chatClient);
        relevanceScorerService = new RelevanceScorerService(chatClientBuilder);
    }

    @Test
    void testScoreJobUsesCorrectResume() throws IOException {
        // Mock Person 1
        Person maya = new Person();
        maya.setName("Maya");
        maya.setResumePath("MayaResume.txt");

        // Mock Person 2
        Person emily = new Person();
        emily.setName("Emily");
        emily.setResumePath("EmilyCV.txt");

        // Mock Job
        JobListing job = new JobListing();
        job.setTitle("Java Dev");
        job.setDescription("Spring Boot expert");

        // Mock Matches
        JobMatch mayaMatch = new JobMatch();
        mayaMatch.setPerson(maya);
        mayaMatch.setJobListing(job);

        JobMatch emilyMatch = new JobMatch();
        emilyMatch.setPerson(emily);
        emilyMatch.setJobListing(job);

        // Mock ChatClient chain
        when(chatClient.prompt(anyString())).thenReturn(requestSpec);
        when(requestSpec.call()).thenReturn(responseSpec);
        when(responseSpec.content()).thenReturn("SCORE: 80\nREASON: Good fit.");

        // We can't easily mock Files.readString, but we can verify the prompt contains the person's name
        // as a proxy for the service using the correct Person context.
        
        relevanceScorerService.scoreJob(mayaMatch);
        ArgumentCaptor<String> promptCaptor = ArgumentCaptor.forClass(String.class);
        verify(chatClient).prompt(promptCaptor.capture());
        assertTrue(promptCaptor.getValue().contains("Maya"));

        reset(chatClient);
        when(chatClient.prompt(anyString())).thenReturn(requestSpec);

        relevanceScorerService.scoreJob(emilyMatch);
        verify(chatClient).prompt(promptCaptor.capture());
        assertTrue(promptCaptor.getValue().contains("Emily"));
    }
}
