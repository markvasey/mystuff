package com.example.housepricemonitor.service;

import com.example.housepricemonitor.model.PropertyDetail;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

public class EpcServiceTest {

    private MockWebServer mockWebServer;
    private EpcService epcService;

    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();

        WebClient.Builder webClientBuilder = WebClient.builder();
        epcService = new EpcService(webClientBuilder);
        
        ReflectionTestUtils.setField(epcService, "baseUrl", mockWebServer.url("/").toString());
        ReflectionTestUtils.setField(epcService, "apiKey", "test-key");
        ReflectionTestUtils.setField(epcService, "apiEmail", "test@test.com");
    }

    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }

    @Test
    void testFetchPropertyDetailSuccess() {
        String jsonResponse = "{\"rows\":[" +
                "{\"address\":\"1 Main St\",\"total-floor-area\":100.5,\"number-habitable-rooms\":5,\"property-age-band\":\"Pre-1900\",\"built-form\":\"Mid-Terrace\"}" +
                "]}";

        mockWebServer.enqueue(new MockResponse()
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        Optional<PropertyDetail> detail = epcService.fetchPropertyDetail("KT4 1AA", "1 Main St");

        assertTrue(detail.isPresent());
        assertEquals(5, detail.get().getHabitableRooms());
        assertEquals(100.5, detail.get().getTotalFloorArea().doubleValue());
    }

    @Test
    void testFetchPropertyDetailNoMatch() {
        String jsonResponse = "{\"rows\":[" +
                "{\"address\":\"2 Main St\",\"total-floor-area\":100.5,\"number-habitable-rooms\":5}" +
                "]}";

        mockWebServer.enqueue(new MockResponse()
                .setBody(jsonResponse)
                .addHeader("Content-Type", "application/json"));

        Optional<PropertyDetail> detail = epcService.fetchPropertyDetail("KT4 1AA", "1 Main St");

        assertFalse(detail.isPresent());
    }

    @Test
    void testFetchPropertyDetailError() {
        mockWebServer.enqueue(new MockResponse().setResponseCode(401));

        Optional<PropertyDetail> detail = epcService.fetchPropertyDetail("KT4 1AA", "1 Main St");

        assertFalse(detail.isPresent());
    }
}
