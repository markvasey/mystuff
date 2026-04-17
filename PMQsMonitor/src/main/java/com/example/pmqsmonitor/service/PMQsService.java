package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.repository.UtteranceRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
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
        log.info("Polling TheyWorkForYou for new PMQs...");
        twfyClient.getPMQs()
                .doOnNext(rows -> {
                    if (rows != null) {
                        processRows(rows);
                    }
                })
                .block();
    }

    private void processRows(List<TWFYClient.TWFYRow> rows) {
        log.info("Processing {} rows from API", rows.size());
        
        try {
            List<TWFYClient.TWFYRow> sortedRows = rows.stream()
                    .sorted(Comparator.comparing((TWFYClient.TWFYRow r) -> r.hdate)
                            .thenComparing(r -> r.htime, Comparator.nullsLast(Comparator.naturalOrder())))
                    .collect(Collectors.toList());

            Utterance lastQuestion = null;

            for (TWFYClient.TWFYRow row : sortedRows) {
                if (row.body == null) continue;

                // 1. FILTER FIRST: Only process PMQs
                String title = row.title != null ? row.title : 
                              (row.parent != null ? row.parent.body : "");
                
                boolean isPMQ = row.listurl.contains("s=Prime+Minister%27s+Questions");

                if (!isPMQ) {
                    log.debug("Skipping non-PMQ row: GID={}, Title={}", row.gid, title);
                    continue;
                }
                
                log.debug("Processing valid PMQ row: GID={}, Title={}", row.gid, title);

                // 2. MAP & SAVE
                String gid = row.gid;
                Optional<Utterance> existing = utteranceRepository.findByExternalId(gid);
                
                Utterance utterance;
                if (existing.isPresent()) {
                    utterance = existing.get();
                    updateMetadata(utterance, row);
                    utterance = utteranceRepository.save(utterance);
                    log.debug("Updated PMQ record: {}", gid);
                } else {
                    utterance = mapToUtterance(row);
                    utterance.setType(utterance.isStarmer() || utterance.isRepresentative() ? "answer" : "question");
                    utterance = utteranceRepository.save(utterance);
                    log.debug("Saved new PMQ utterance: {}, Starmer: {}", gid, utterance.isStarmer());
                }

                // 3. TRIGGER ANALYSIS
                if (utterance.isStarmer() || utterance.isRepresentative()) {
                    if (row.parent != null) {
                        Utterance question = utteranceRepository.findByExternalId(row.parent.gid)
                                .orElseGet(() -> {
                                    Utterance q = mapToUtterance(row.parent);
                                    q.setType("question");
                                    return utteranceRepository.save(q);
                                });
                        log.info("Analyzing answer to parent question from {}", question.getSpeakerName());
                        analysisService.analyzeUtterance(question, utterance);
                    } else if (lastQuestion != null && !"answer".equals(lastQuestion.getType())) {
                        log.info("Analyzing answer to chronological question from {}", lastQuestion.getSpeakerName());
                        analysisService.analyzeUtterance(lastQuestion, utterance);
                    }
                } else {
                    lastQuestion = utterance;
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
                        .map(o -> o.position + (o.dept != null ? " (" + o.dept + ")" : ""))
                        .collect(Collectors.joining(", ")));
            }
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
                        .map(o -> o.position + (o.dept != null ? " (" + o.dept + ")" : ""))
                        .collect(Collectors.joining(", ")));
            }
        }
        
        if (row.parent != null) {
            u.setParentBody(row.parent.body);
        }
    }

    public List<Utterance> getLatestPMQs() {
        return utteranceRepository.findAll().stream()
                .filter(u -> u.getDateTime() != null && u.getDateTime().getDayOfWeek() == java.time.DayOfWeek.WEDNESDAY)
                .sorted(Comparator.comparing(Utterance::getDateTime).reversed())
                .limit(100)
                .collect(Collectors.toList());
    }
}
