package com.whatsapp.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;

import java.util.List;

import static org.mockito.Mockito.*;

public class WhatsAppOrchestratorTest {

    @Mock
    private WhatsAppClient mockClient;
    
    @Mock
    private AccountManager mockAccountManager;

    private WhatsAppOrchestrator orchestrator;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.openMocks(this);
        orchestrator = new WhatsAppOrchestrator(mockClient, mockAccountManager);
    }

    @Test
    public void testStartServiceSequence() throws Exception {
        when(mockAccountManager.getAccounts()).thenReturn(List.of(
            new AccountManager.Account("Test", "12345")
        ));
        
        orchestrator.startService("test-session");

        verify(mockClient).initialize();
        verify(mockClient).createConnectionOptions("test-session");
        verify(mockClient).start(any(), eq("12345"));
        verify(mockClient).awaitHandshake(anyLong(), any());
        verify(mockClient).awaitLogin(anyLong(), any());
    }

    @Test
    public void testStopCallsDisconnect() {
        orchestrator.stop();
        verify(mockClient).disconnect();
    }
}
