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
    public void processRows(List<TWFYClient.TWFYRow> rows) {
        log.info("Processing {} rows from full debate transcript", rows.size());
        
        try {
            List<TWFYClient.TWFYRow> sortedRows = rows.stream()
                    .sorted(Comparator.comparing((TWFYClient.TWFYRow r) -> r.hdate != null ? r.hdate : "")
                            .thenComparing(r -> r.htime, Comparator.nullsLast(Comparator.naturalOrder())))
                    .collect(Collectors.toList());

            Utterance lastQuestion = null;

            for (TWFYClient.TWFYRow row : sortedRows) {
                if (row.body == null || row.body.trim().isEmpty()) continue;

                String gid = row.gid;
                Optional<Utterance> existing = utteranceRepository.findByExternalId(gid);
                
                Utterance utterance;
                if (existing.isPresent()) {
                    utterance = existing.get();
                    updateMetadata(utterance, row);
                    utterance.setType(utterance.isStarmer() || utterance.isRepresentative() ? "answer" : "question");
                    utterance = utteranceRepository.save(utterance);
                } else {
                    utterance = mapToUtterance(row);
                    utterance.setType(utterance.isStarmer() || utterance.isRepresentative() ? "answer" : "question");
                    utterance = utteranceRepository.save(utterance);
                    log.debug("Saved new utterance: {}, Starmer: {}", gid, utterance.isStarmer());
                }

                // Trigger Analysis logic
                if (utterance.isStarmer() || utterance.isRepresentative()) {
                    if (lastQuestion != null && !"answer".equals(lastQuestion.getType())) {
                        String qName = lastQuestion.getSpeakerName() != null ? lastQuestion.getSpeakerName() : "Unknown MP";
                        //log.info("Analyzing answer to question from {}", qName);
                        //analysisService.analyzeUtterance(lastQuestion, utterance);
                    }
                } else {
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
        
        // Fix: Default to Oral questions for PMQs
        u.setDebateType(row.debateType != null ? row.debateType : "Oral questions");
        
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
        
        // Fix: Ensure parentBody is set
        if (row.parent != null) {
            u.setParentBody(row.parent.body);
        } else if (u.getParentBody() == null) {
            u.setParentBody("Prime Minister: Engagements");
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
        
        // Fix: Default to Oral questions
        u.setDebateType(row.debateType != null ? row.debateType : "Oral questions");
        
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
        
        // Fix: Ensure parentBody is set
        if (row.parent != null) {
            u.setParentBody(row.parent.body);
        } else if (u.getParentBody() == null) {
            u.setParentBody("Prime Minister: Engagements");
        }
    }

    public List<java.time.LocalDate> getAvailableDates() {
        return utteranceRepository.findDistinctDates().stream()
                .map(java.sql.Date::toLocalDate)
                .collect(Collectors.toList());
    }

    public List<Utterance> getUtterancesByDate(java.time.LocalDate date, boolean hideSpeaker) {
        return utteranceRepository.findAll().stream()
                .filter(u -> u.getDateTime() != null && u.getDateTime().toLocalDate().equals(date))
                .filter(u -> u.getSpeakerId() != null) 
                .filter(u -> !hideSpeaker || (u.getParty() != null && !u.getParty().equalsIgnoreCase("Speaker")))
                .sorted(Comparator.comparing(Utterance::getDateTime))
                .collect(Collectors.toList());
    }
}
