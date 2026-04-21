package com.example.pmqsmonitor.service;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
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
                                .responseTimeout(java.time.Duration.ofSeconds(60)))) 
                .codecs(configurer -> configurer
                        .defaultCodecs()
                        .maxInMemorySize(100 * 1024 * 1024)) 
                .build();
        this.apiKey = apiKey;
    }

    public Mono<String> getXmlDirectoryIndex() {
        return webClient.get()
                .uri("https://www.theyworkforyou.com/pwdata/scrapedxml/debates/")
                .retrieve()
                .bodyToMono(String.class);
    }

    public Mono<String> getRawXmlFile(String filename) {
        return webClient.get()
                .uri("https://www.theyworkforyou.com/pwdata/scrapedxml/debates/" + filename)
                .retrieve()
                .bodyToMono(String.class);
    }

    public Mono<List<TWFYRow>> searchForPMQsHeader(String date) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("getDebates")
                        .queryParam("type", "commons")
                        .queryParam("date", date)
                        .queryParam("search", "Engagements")
                        .queryParam("output", "json")
                        .queryParam("key", apiKey)
                        .build())
                .retrieve()
                .bodyToMono(TWFYDebateResponse.class)
                .map(res -> res != null && res.rows != null ? res.rows : List.<TWFYRow>of());
    }

    /**
     * Get the full debate transcript for a specific GID.
     * Use bodyToMono(TWFYRow[].class) to correctly handle raw JSON arrays.
     */
    public Mono<List<TWFYRow>> getFullDebateByGid(String gid) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("getDebates")
                        .queryParam("type", "commons")
                        .queryParam("gid", gid)
                        .queryParam("output", "json")
                        .queryParam("key", apiKey)
                        .build())
                .retrieve()
                .bodyToMono(TWFYRow[].class)
                .map(Arrays::asList);
    }

    public Mono<List<TWFYRow>> getPMQs() {
        return searchForPMQsHeader("2026-04-15");
    }

    public static class TWFYDebateResponse {
        @JsonProperty("rows")
        public List<TWFYRow> rows;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class TWFYRow {
        public String gid;
        @JsonProperty("hdate")
        public String hdate;
        @JsonProperty("htime")
        public String htime;
        public String body;
        public String listurl;
        @JsonProperty("debate_type")
        public String debateType;
        public String title;
        
        public SpeakerInfo speaker;
        public TWFYRow parent;

        @JsonProperty("speaker")
        public void setSpeaker(Object speaker) {
            if (speaker instanceof java.util.Map) {
                com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
                this.speaker = mapper.convertValue(speaker, SpeakerInfo.class);
            } else {
                this.speaker = null;
            }
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        public static class SpeakerInfo {
            @JsonProperty("person_id")
            public String personId;
            public String name;
            public String party;
            public String house;
            public List<OfficeInfo> office;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        public static class OfficeInfo {
            public String position;
            public String dept;
        }

        public String getGid() { return gid; }
        public String getHdate() { return hdate; }
        public String getHtime() { return htime; }
        public String getBody() { return body; }
        public String getListurl() { return listurl; }
        public String getDebateType() { return debateType; }
        public String getTitle() { return title; }
        public SpeakerInfo getSpeaker() { return speaker; }
        public TWFYRow getParent() { return parent; }
    }
}
