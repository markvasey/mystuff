package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.UtteranceRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.TemporalAdjusters;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class PMQsService {

    private static final Logger log = LoggerFactory.getLogger(PMQsService.class);

    private final TWFYClient twfyClient;
    private final UtteranceRepository utteranceRepository;
    private final AnalysisService analysisService;

    public PMQsService(TWFYClient twfyClient, 
                       UtteranceRepository utteranceRepository, 
                       AnalysisService analysisService) {
        this.twfyClient = twfyClient;
        this.utteranceRepository = utteranceRepository;
        this.analysisService = analysisService;
    }

    @Transactional
    public void pollNow() {
        // Step 1: Find the most recent Wednesday
        LocalDate lastWednesday = LocalDate.now().with(TemporalAdjusters.previousOrSame(DayOfWeek.WEDNESDAY));
        String dateStr = lastWednesday.format(DateTimeFormatter.ISO_LOCAL_DATE);
        
        log.info("Polling TheyWorkForYou for PMQs header on {}", dateStr);
        
        twfyClient.searchForPMQsHeader(dateStr)
                .flatMap(rows -> {
                    // Step 2: Look for the specific "Prime Minister" engagements entry to get the GID
                    Optional<TWFYClient.TWFYRow> pmqsHeader = rows.stream()
                            .filter(r -> r.body != null && r.body.contains("Prime Minister"))
                            .findFirst();

                    if (pmqsHeader.isPresent()) {
                        String gid = pmqsHeader.get().gid;
                        log.info("Found PMQs session GID: {}. Fetching full debate...", gid);
                        return twfyClient.getFullDebateByGid(gid);
                    } else {
                        log.warn("Could not find PMQs header (body: 'Prime Minister') for {}", dateStr);
                        return reactor.core.publisher.Mono.just(List.<TWFYClient.TWFYRow>of());
                    }
                })
                .doOnNext(fullDebate -> {
                    if (!fullDebate.isEmpty()) {
                        processRows(fullDebate);
                    }
                })
                .block();
    }

    @Transactional
    public void pollNowWithGid(String gid, List<TWFYClient.TWFYRow> rows) {
        log.info("Manual poll triggered for GID: {} ({} rows)", gid, rows.size());
        processRows(rows);
    }

    private void processRows(List<TWFYClient.TWFYRow> rows) {
        log.info("Processing {} rows from full debate transcript", rows.size());
        
        try {
            List<TWFYClient.TWFYRow> sortedRows = rows.stream()
                    .sorted(Comparator.comparing((TWFYClient.TWFYRow r) -> r.hdate != null ? r.hdate : "")
                            .thenComparing(r -> r.htime, Comparator.nullsLast(Comparator.naturalOrder())))
                    .collect(Collectors.toList());

            Utterance lastQuestion = null;

            for (TWFYClient.TWFYRow row : sortedRows) {
                if (row.body == null || row.body.trim().isEmpty()) continue;

                log.info("PMQsService.processRows: {} {}", row.speaker== null ? "" : row.speaker.name, row.body);

                String gid = row.gid;
                Optional<Utterance> existing = utteranceRepository.findByExternalId(gid);
                
                Utterance utterance;
                if (existing.isPresent()) {
                    utterance = existing.get();
                    updateMetadata(utterance, row);
                    utterance = utteranceRepository.save(utterance);
                } else {
                    utterance = mapToUtterance(row);
                    // Determine type: default to question, unless it's Starmer
                    utterance.setType(utterance.isStarmer() || utterance.isRepresentative() ? "answer" : "question");
                    utterance = utteranceRepository.save(utterance);
                    log.debug("Saved new utterance: {}, Starmer: {}", gid, utterance.isStarmer());
                }

                // Trigger Analysis logic
                if (utterance.isStarmer() || utterance.isRepresentative()) {
                    if (lastQuestion != null && !"answer".equals(lastQuestion.getType())) {
                        String qName = lastQuestion.getSpeakerName() != null ? lastQuestion.getSpeakerName() : "Unknown MP";
                        log.info("Analyzing answer to question from {}", qName);
                        analysisService.analyzeUtterance(lastQuestion, utterance);
                    }
                } else {
                    // Logic: If it's not Starmer and it's not procedural, it's a question
                    if (utterance.getSpeakerName() != null && !utterance.getSpeakerName().contains("Speaker")) {
                        lastQuestion = utterance;
                    }
                }
            }
        } catch (Exception e) {
            log.error("Error during row processing: {}", e.getMessage(), e);
        }
    }

    private Utterance mapToUtterance(TWFYClient.TWFYRow row) {
        Utterance u = new Utterance();
        u.setExternalId(row.gid);
        u.setHdate(row.hdate);
        u.setHtime(row.htime);
        u.setListurl(row.listurl != null ? "https://www.theyworkforyou.com" + row.listurl : null);
        u.setDebateType(row.debateType);
        
        if (row.speaker != null) {
            u.setSpeakerId(row.speaker.personId);
            u.setSpeakerName(row.speaker.name);
            u.setParty(row.speaker.party);
            u.setHouse(row.speaker.house);
            
            if (row.speaker.office != null && !row.speaker.office.isEmpty()) {
                u.setOffice(row.speaker.office.stream()
                        .map(o -> o.position + (o.dept != null && !o.dept.isEmpty() ? " (" + o.dept + ")" : ""))
                        .collect(Collectors.joining(", ")));
            }
        }
        
        if (u.getSpeakerName() == null && row.title != null) {
             u.setSpeakerName(row.title);
        }
        
        if (row.parent != null) {
            u.setParentBody(row.parent.body);
        }
        
        u.setText(row.body);
        
        try {
            LocalDate date = (row.hdate != null) ? LocalDate.parse(row.hdate, DateTimeFormatter.ISO_LOCAL_DATE) : LocalDate.now();
            LocalTime time = (row.htime != null && !row.htime.isEmpty()) ? LocalTime.parse(row.htime) : LocalTime.MIDNIGHT;
            u.setDateTime(LocalDateTime.of(date, time));
        } catch (Exception e) {
            u.setDateTime(LocalDateTime.now());
        }

        u.setStarmer("25353".equals(u.getSpeakerId()) 
                   || "Keir Starmer".equalsIgnoreCase(u.getSpeakerName())
                   || "The Prime Minister".equalsIgnoreCase(u.getSpeakerName()));
        
        u.setRepresentative(u.getText() != null && (u.getText().toLowerCase().contains("on behalf of the prime minister") 
                           || u.getText().toLowerCase().contains("representing the prime minister")));

        return u;
    }

    private void updateMetadata(Utterance u, TWFYClient.TWFYRow row) {
        u.setHdate(row.hdate);
        u.setHtime(row.htime);
        u.setListurl(row.listurl != null ? "https://www.theyworkforyou.com" + row.listurl : u.getListurl());
        u.setDebateType(row.debateType);
        
        if (row.speaker != null) {
            u.setSpeakerId(row.speaker.personId);
            u.setSpeakerName(row.speaker.name);
            u.setParty(row.speaker.party);
            u.setHouse(row.speaker.house);
            
            if (row.speaker.office != null && !row.speaker.office.isEmpty()) {
                u.setOffice(row.speaker.office.stream()
                        .map(o -> o.position + (o.dept != null && !o.dept.isEmpty() ? " (" + o.dept + ")" : ""))
                        .collect(Collectors.joining(", ")));
            }
        }
        
        if (row.parent != null && u.getParentBody() == null) {
            u.setParentBody(row.parent.body);
        }
    }

    public List<Utterance> getLatestPMQs() {
        return utteranceRepository.findAll().stream()
                .filter(u -> u.getDateTime() != null && u.getDateTime().getDayOfWeek() == DayOfWeek.WEDNESDAY)
                .sorted(Comparator.comparing(Utterance::getDateTime).reversed())
                .limit(200)
                .collect(Collectors.toList());
    }
}
