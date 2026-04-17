package com.example.pmqsmonitor.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

@Service
public class TWFYClient {

    private static final Logger log = LoggerFactory.getLogger(TWFYClient.class);
    private final WebClient webClient;
    private final String apiKey;

    public TWFYClient(WebClient.Builder webClientBuilder, 
                      @Value("${app.twfy.base-url}") String baseUrl,
                      @Value("${app.twfy.api-key}") String apiKey) {
        this.webClient = webClientBuilder
                .baseUrl(baseUrl)
                .clientConnector(new org.springframework.http.client.reactive.ReactorClientHttpConnector(
                        reactor.netty.http.client.HttpClient.create()
                                .responseTimeout(java.time.Duration.ofSeconds(30)))) // 30s timeout for large payload
                .codecs(configurer -> configurer
                        .defaultCodecs()
                        .maxInMemorySize(50 * 1024 * 1024)) // 50MB buffer
                .build();
        this.apiKey = apiKey;
    }

    public Mono<List<TWFYRow>> getPMQs() {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("getHansard")
                        .queryParam("search", "Prime Minister's Questions")
                        .queryParam("num", "1000") // Get more than 20 results
                        .queryParam("output", "json")
                        .queryParam("key", apiKey)
                        .build())
                .retrieve()
                .bodyToMono(String.class)
                .map(json -> {
                    //log.debug("RAW API JSON: {}", json);
                    try {
                        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper()
                                .configure(com.fasterxml.jackson.databind.DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
                        TWFYDebateResponse res = mapper.readValue(json, TWFYDebateResponse.class);
                        if (res != null && res.rows != null) {
                            log.debug("Successfully parsed {} rows from API JSON", res.rows.size());
                            if (!res.rows.isEmpty()) {
                                int count = 0;
                                for(TWFYRow row : res.rows) {
                                    count++;
                                    log.debug("Mapping Check (Row {}): GID={}, HDate={}, HTime={}, SpeakerName={}, SpeakerParty={}, Body length={}, listURL={}, title={}, debateType={}, parent={}",
                                            count, row.gid, row.hdate, row.htime,
                                            (row.speaker != null ? row.speaker.name : "null"),
                                            (row.speaker != null ? row.speaker.party : "null"),
                                            (row.body != null ? row.body.length() : 0), row.listurl, row.title, row.debateType, row.getParent().getBody());
                                }
                            }
                            return res.rows;
                        }
                    } catch (Exception e) {
                        log.error("JSON Parsing Error: {}", e.getMessage(), e);
                    }
                    return List.<TWFYRow>of();
                });
    }

    public static class TWFYDebateResponse {
        @JsonProperty("rows")
        public List<TWFYRow> rows;
    }

    public static class TWFYRow {
        @JsonProperty("gid")
        public String gid;
        @JsonProperty("hdate")
        public String hdate;
        @JsonProperty("htime")
        public String htime;
        @JsonProperty("body")
        public String body;
        @JsonProperty("listurl")
        public String listurl;
        @JsonProperty("debate_type")
        public String debateType;
        @JsonProperty("title")
        public String title;
        
        @JsonProperty("speaker")
        public SpeakerInfo speaker;
        @JsonProperty("parent")
        public TWFYRow parent;

        public static class SpeakerInfo {
            @JsonProperty("person_id")
            public String personId;
            @JsonProperty("name")
            public String name;
            @JsonProperty("party")
            public String party;
            @JsonProperty("house")
            public String house;
            @JsonProperty("office")
            public List<OfficeInfo> office;
        }

        public static class OfficeInfo {
            @JsonProperty("position")
            public String position;
            @JsonProperty("dept")
            public String dept;
        }

        // Keep getters for PMQsService
        public String getGid() { return gid; }
        public String getDate() { return hdate; }
        public String getTime() { return htime; }
        public String getBody() { return body; }
        public String getListurl() { return listurl; }
        public String getDebateType() { return debateType; }
        public String getTitle() { return title; }
        public SpeakerInfo getSpeaker() { return speaker; }
        public TWFYRow getParent() { return parent; }
    }
}
