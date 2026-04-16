package com.example.housepricemonitor.service;

import com.example.housepricemonitor.model.PropertyTransaction;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class LandRegistryServiceTest {

    private MockWebServer mockWebServer;
    private LandRegistryService landRegistryService;

    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();

        WebClient.Builder webClientBuilder = WebClient.builder();
        landRegistryService = new LandRegistryService(webClientBuilder);
        
        ReflectionTestUtils.setField(landRegistryService, "apiUrl", mockWebServer.url("/query").toString());
    }

    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }

    @Test
    void testFetchTransactionsSuccess() {
        String jsonResponse = "{\"head\":{\"vars\":[\"trans\",\"price\",\"date\",\"postcode\",\"paon\",\"type\"]},\"results\":{\"bindings\":[" +
                "{\"trans\":{\"value\":\"tx1\"},\"price\":{\"value\":\"100000\"},\"date\":{\"value\":\"2023-01-01\"},\"postcode\":{\"value\":\"KT4 1AA\"},\"paon\":{\"value\":\"1\"},\"type\":{\"value\":\"Detached\"}}" +
                "]}}";

        mockWebServer.enqueue(new MockResponse()
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        List<PropertyTransaction> transactions = landRegistryService.fetchTransactions(Arrays.asList("KT4"), LocalDate.now().minusMonths(1));

        assertNotNull(transactions);
        assertEquals(1, transactions.size());
        assertEquals("tx1", transactions.get(0).getTransactionId());
        assertEquals("1 ", transactions.get(0).getAddress()); // paon + street (empty here)
    }

    @Test
    void testFetchTransactionsEmptyResponse() {
        String jsonResponse = "{\"head\":{\"vars\":[]},\"results\":{\"bindings\":[]}}";

        mockWebServer.enqueue(new MockResponse()
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        List<PropertyTransaction> transactions = landRegistryService.fetchTransactions(Arrays.asList("KT4"), LocalDate.now().minusMonths(1));

        assertNotNull(transactions);
        assertTrue(transactions.isEmpty());
    }

    @Test
    void testFetchTransactionsError() {
        mockWebServer.enqueue(new MockResponse().setResponseCode(500));

        assertThrows(Exception.class, () -> {
            landRegistryService.fetchTransactions(Arrays.asList("KT4"), LocalDate.now().minusMonths(1));
        });
    }
}
