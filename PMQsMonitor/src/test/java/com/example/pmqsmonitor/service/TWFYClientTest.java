package com.example.pmqsmonitor.service;

import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.web.reactive.function.client.WebClient;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class TWFYClientTest {

    private MockWebServer mockWebServer;
    private TWFYClient twfyClient;

    @BeforeEach
    void setUp() throws IOException {
        mockWebServer = new MockWebServer();
        mockWebServer.start();

        WebClient.Builder webClientBuilder = WebClient.builder();
        twfyClient = new TWFYClient(webClientBuilder, mockWebServer.url("/").toString(), "test-key");
    }

    @AfterEach
    void tearDown() throws IOException {
        mockWebServer.shutdown();
    }

    @Test
    void testGetFullDebateByGid_ShortArray() {
        // Arrange
        String mockJsonResponse = """
        [
            {
                "epobject_id": "29859458",
                "htype": "10",
                "gid": "2026-01-28d.890.3",
                "hpos": "80",
                "section_id": "0",
                "subsection_id": "0",
                "hdate": "2026-01-28",
                "htime": null,
                "source_url": "",
                "major": "1",
                "minor": "0",
                "colnum": "890",
                "body": "Prime Minister",
                "contentcount": "0",
                "listurl": "/debates/?id=2026-01-28d.890.3",
                "commentsurl": "/debates/?id=2026-01-28d.890.3"
            },
            {
                "epobject_id": "29859459",
                "htype": "13",
                "gid": "2026-01-28d.890.4",
                "hpos": "81",
                "section_id": "29859458",
                "subsection_id": "29859458",
                "hdate": "2026-01-28",
                "htime": null,
                "source_url": "",
                "major": "1",
                "minor": "0",
                "colnum": "890",
                "person_id": "0",
                "body": "<p class=\\"italic\\" pid=\\"d890.4/1\\">The Prime Minister was asked&#8212;</p>",
                "listurl": "/debates/?id=2026-01-28d.890.3#g890.4",
                "commentsurl": "/debates/?id=2026-01-28d.890.4",
                "speaker": [],
                "totalcomments": "0",
                "comment": []
            }
        ]
        """;

        mockWebServer.enqueue(new MockResponse()
                .setBody(mockJsonResponse)
                .addHeader("Content-Type", "application/json"));

        // Act
        List<TWFYClient.TWFYRow> rows = twfyClient.getFullDebateByGid("2026-01-28d.890.3").block();

        // Assert
        assertNotNull(rows);
        assertEquals(2, rows.size());
        assertEquals("2026-01-28d.890.3", rows.get(0).gid);
        assertEquals("Prime Minister", rows.get(0).body);
        assertEquals("2026-01-28d.890.4", rows.get(1).gid);
    }

    @Test
    void testGetFullDebateByGid_FullSessionArray() {
        // Arrange
        String mockJsonResponse = """
        [
            {
                "epobject_id": "29996516",
                "htype": "10",
                "gid": "2026-03-25e.290.0",
                "hdate": "2026-03-25",
                "body": "Prime Minister"
            },
            {
                "epobject_id": "29996518",
                "htype": "11",
                "gid": "2026-03-25e.290.2",
                "hdate": "2026-03-25",
                "body": "Engagements"
            },
            {
                "epobject_id": "29996519",
                "htype": "12",
                "gid": "2026-03-25e.290.3",
                "hdate": "2026-03-25",
                "speaker": {
                    "name": "Cat Smith",
                    "party": "Labour",
                    "person_id": "25432"
                },
                "body": "<p>If he will list his official engagements...</p>"
            },
            {
                "epobject_id": "29996520",
                "htype": "12",
                "gid": "2026-03-25e.290.4",
                "hdate": "2026-03-25",
                "speaker": {
                    "name": "Keir Starmer",
                    "party": "Labour",
                    "person_id": "25353"
                },
                "body": "<p>An attack on Britain's Jewish community...</p>"
            }
        ]
        """;

        mockWebServer.enqueue(new MockResponse()
                .setBody(mockJsonResponse)
                .addHeader("Content-Type", "application/json"));

        // Act
        List<TWFYClient.TWFYRow> rows = twfyClient.getFullDebateByGid("2026-03-25e.290.0").block();

        // Assert
        assertNotNull(rows);
        assertEquals(4, rows.size());
        assertEquals("Keir Starmer", rows.get(3).speaker.name);
        assertEquals("25353", rows.get(3).speaker.personId);
    }
}
