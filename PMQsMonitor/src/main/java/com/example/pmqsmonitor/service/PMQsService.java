package com.example.pmqsmonitor.service;

import com.example.pmqsmonitor.model.AnalysisResult;
import com.example.pmqsmonitor.model.Utterance;
import com.example.pmqsmonitor.model.SessionSummary;
import com.example.pmqsmonitor.repository.UtteranceRepository;
import com.example.pmqsmonitor.repository.AnalysisResultRepository;
import com.example.pmqsmonitor.repository.SessionSummaryRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class PMQsService {

    private static final Logger log = LoggerFactory.getLogger(PMQsService.class);

    private final TWFYClient twfyClient;
    private final UtteranceRepository utteranceRepository;
    private final AnalysisService analysisService;
    private final AnalysisResultRepository analysisResultRepository;
    private final SessionSummaryRepository sessionSummaryRepository;

    public PMQsService(TWFYClient twfyClient, 
                       UtteranceRepository utteranceRepository, 
                       AnalysisService analysisService,
                       AnalysisResultRepository analysisResultRepository,
                       SessionSummaryRepository sessionSummaryRepository) {
        this.twfyClient = twfyClient;
        this.utteranceRepository = utteranceRepository;
        this.analysisService = analysisService;
        this.analysisResultRepository = analysisResultRepository;
        this.sessionSummaryRepository = sessionSummaryRepository;
    }

    @Transactional
    public void processRows(List<TWFYClient.TWFYRow> rows) {
        log.info("Processing {} rows from full debate transcript", rows.size());
        
        try {
            List<TWFYClient.TWFYRow> sortedRows = rows.stream()
                    .sorted(Comparator.comparing((TWFYClient.TWFYRow r) -> r.hdate != null ? r.hdate : "")
                            .thenComparing(r -> r.htime, Comparator.nullsLast(Comparator.naturalOrder())))
                    .collect(Collectors.toList());

            List<TWFYClient.TWFYRow> mergedRows = new ArrayList<>();
            TWFYClient.TWFYRow currentMerged = null;

            for (TWFYClient.TWFYRow row : sortedRows) {
                if (row.body == null || row.body.trim().isEmpty()) continue;
                
                String currentSpeakerId = (row.speaker != null) ? row.speaker.personId : ("unknown-" + row.gid);

                if (currentMerged == null) {
                    currentMerged = row;
                    mergedRows.add(currentMerged);
                } else {
                    String lastSpeakerId = (currentMerged.speaker != null) ? currentMerged.speaker.personId : ("unknown-" + currentMerged.gid);
                    
                    if (currentSpeakerId.equals(lastSpeakerId)) {
                        currentMerged.body += "<br/><br/>" + row.body;
                    } else {
                        currentMerged = row;
                        mergedRows.add(currentMerged);
                    }
                }
            }

            for (TWFYClient.TWFYRow row : mergedRows) {
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
        
        if (row.parent != null) {
            u.setParentBody(row.parent.body);
        } else {
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
        
        if (row.parent != null) {
            u.setParentBody(row.parent.body);
        }
    }

    public List<java.time.LocalDate> getAvailableDates() {
        return utteranceRepository.findDistinctDates().stream()
                .map(java.sql.Date::toLocalDate)
                .collect(Collectors.toList());
    }

    public List<Utterance> getUtterancesByDate(java.time.LocalDate date, boolean hideSpeaker) {
        List<Utterance> rawList = utteranceRepository.findAll().stream()
                .filter(u -> u.getDateTime() != null && u.getDateTime().toLocalDate().equals(date))
                .filter(u -> u.getSpeakerId() != null) 
                .filter(u -> !hideSpeaker || (u.getParty() != null && !u.getParty().equalsIgnoreCase("Speaker")))
                .sorted(Comparator.comparing(Utterance::getDateTime))
                .collect(Collectors.toList());

        return mergeConsecutiveUtterances(rawList);
    }

    public SessionSummary getSessionSummary(java.time.LocalDate date, boolean hideSpeaker) {
        return sessionSummaryRepository.findBySessionDate(date).orElse(null);
    }

    public Map<String, Long> getSentimentCounts(java.time.LocalDate date, boolean hideSpeaker) {
        List<Utterance> utterances = getUtterancesByDate(date, hideSpeaker);
        return utterances.stream()
                .map(Utterance::getAnalysisResult)
                .filter(java.util.Objects::nonNull)
                .collect(Collectors.groupingBy(AnalysisResult::getSentiment, Collectors.counting()));
    }

    public Map<String, Long> getDiversionCounts(java.time.LocalDate date, boolean hideSpeaker) {
        List<Utterance> utterances = getUtterancesByDate(date, hideSpeaker);
        return utterances.stream()
                .map(Utterance::getAnalysisResult)
                .filter(java.util.Objects::nonNull)
                .flatMap(r -> r.getDiversionTactics().stream())
                .collect(Collectors.groupingBy(t -> t, Collectors.counting()));
    }

    private void calculateAndSaveSummary(java.time.LocalDate date, boolean hideSpeaker) {
        log.info("Calculating final session summary for {}...", date);
        List<Utterance> utterances = getUtterancesByDate(date, hideSpeaker);
        List<AnalysisResult> results = utterances.stream()
                .map(Utterance::getAnalysisResult)
                .filter(java.util.Objects::nonNull)
                .collect(Collectors.toList());

        if (results.isEmpty()) return;

        SessionSummary summary = sessionSummaryRepository.findBySessionDate(date)
                .orElse(new SessionSummary());
        
        summary.setSessionDate(date);
        summary.setTotalAnalyzed(results.size());
        summary.setAvgCompleteness(results.stream().mapToInt(AnalysisResult::getCompleteness).average().orElse(0));
        summary.setAvgRelevance(results.stream().mapToInt(AnalysisResult::getRelevance).average().orElse(0));
        summary.setDirectAnswers((int) results.stream().filter(AnalysisResult::isDirectAnswer).count());
        
        List<String> rationales = results.stream().map(AnalysisResult::getRational).collect(Collectors.toList());
        summary.setExecutiveSummary(analysisService.summarizeRationales(rationales));
        summary.setCalculatedAt(LocalDateTime.now());

        sessionSummaryRepository.save(summary);
    }

    @Transactional
    public void analyzeSession(java.time.LocalDate date, boolean hideSpeaker) {
        List<Utterance> utterances = getUtterancesByDate(date, hideSpeaker);
        log.info("Analyzing session for {}: {} merged utterances", date, utterances.size());

        Utterance lastQuestion = null;
        for (Utterance utterance : utterances) {
            if (utterance.isStarmer() || utterance.isRepresentative()) {
                if (lastQuestion != null && "question".equals(lastQuestion.getType())) {
                    log.info("Triggering AI analysis for answer to question from {}", lastQuestion.getSpeakerName());
                    analysisService.analyzeUtterance(lastQuestion, utterance);
                    
                    try { Thread.sleep(2000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                }
            } else if ("question".equals(utterance.getType())) {
                lastQuestion = utterance;
            }
        }
        calculateAndSaveSummary(date, hideSpeaker);
    }

    private List<Utterance> mergeConsecutiveUtterances(List<Utterance> utterances) {
        if (utterances == null || utterances.isEmpty()) {
            return utterances;
        }

        List<Utterance> merged = new ArrayList<>();
        Utterance current = null;

        for (Utterance u : utterances) {
            if (current == null) {
                current = cloneUtterance(u);
                merged.add(current);
            } else if (u.getSpeakerId().equals(current.getSpeakerId())) {
                current.setText(current.getText() + "<br/><br/>" + u.getText());
                if (current.getAnalysisResult() == null && u.getAnalysisResult() != null) {
                    current.setAnalysisResult(u.getAnalysisResult());
                }
            } else {
                current = cloneUtterance(u);
                merged.add(current);
            }
        }
        return merged;
    }

    private Utterance cloneUtterance(Utterance u) {
        Utterance clone = new Utterance();
        clone.setId(u.getId());
        clone.setExternalId(u.getExternalId());
        clone.setSpeakerName(u.getSpeakerName());
        clone.setSpeakerId(u.getSpeakerId());
        clone.setText(u.getText());
        clone.setDateTime(u.getDateTime());
        clone.setType(u.getType());
        clone.setHdate(u.getHdate());
        clone.setHtime(u.getHtime());
        clone.setListurl(u.getListurl());
        clone.setParty(u.getParty());
        clone.setHouse(u.getHouse());
        clone.setOffice(u.getOffice());
        clone.setParentBody(u.getParentBody());
        clone.setDebateType(u.getDebateType());
        clone.setStarmer(u.isStarmer());
        clone.setRepresentative(u.isRepresentative());
        clone.setAnalysisResult(u.getAnalysisResult());
        return clone;
    }
}
