package com.example.housepricemonitor.service;

import com.example.housepricemonitor.model.MonitoredArea;
import com.example.housepricemonitor.model.PropertyDetail;
import com.example.housepricemonitor.model.PropertyTransaction;
import com.example.housepricemonitor.repository.MonitoredAreaRepository;
import com.example.housepricemonitor.repository.PropertyDetailRepository;
import com.example.housepricemonitor.repository.PropertyTransactionRepository;
import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
public class HousePricePoller {

    private static final Logger log = LoggerFactory.getLogger(HousePricePoller.class);

    private final MonitoredAreaRepository monitoredAreaRepository;
    private final PropertyTransactionRepository propertyTransactionRepository;
    private final PropertyDetailRepository propertyDetailRepository;
    private final LandRegistryService landRegistryService;
    private final EpcService epcService;

    public HousePricePoller(MonitoredAreaRepository monitoredAreaRepository,
                            PropertyTransactionRepository propertyTransactionRepository,
                            PropertyDetailRepository propertyDetailRepository,
                            LandRegistryService landRegistryService,
                            EpcService epcService) {
        this.monitoredAreaRepository = monitoredAreaRepository;
        this.propertyTransactionRepository = propertyTransactionRepository;
        this.propertyDetailRepository = propertyDetailRepository;
        this.landRegistryService = landRegistryService;
        this.epcService = epcService;
    }

    @PostConstruct
    public void initAreas() {
        log.info("Initializing monitored areas...");
        if (monitoredAreaRepository.count() == 0) {
            saveArea("TS27", "Hartlepool");
            saveArea("EX39", "Bideford");
            saveArea("KT4", "Kingston-Upon-Thames");
            log.info("Areas initialized: TS27, EX39, KT4");
        } else {
            log.info("Areas already initialized. Total areas: {}", monitoredAreaRepository.count());
        }
    }

    private void saveArea(String district, String name) {
        MonitoredArea area = new MonitoredArea();
        area.setPostcodeDistrict(district);
        area.setName(name);
        monitoredAreaRepository.save(area);
    }

    @Scheduled(cron = "${app.polling.cron}")
    @Transactional
    public void pollNewData() {
        List<MonitoredArea> areas = monitoredAreaRepository.findAll();
        log.info("Starting data poll for {} areas", areas.size());
        
        List<String> districts = areas.stream()
                .map(MonitoredArea::getPostcodeDistrict)
                .collect(Collectors.toList());

        if (districts.isEmpty()) {
            log.warn("No monitored areas found. Skipping poll.");
            return;
        }

        // Fetch last 12 months for initial load.
        LocalDate sinceDate = LocalDate.now().minusMonths(12);
        log.info("Fetching transactions since {} for districts: {}", sinceDate, districts);
        
        try {
            List<PropertyTransaction> transactions = landRegistryService.fetchTransactions(districts, sinceDate);
            log.info("Found {} transactions from Land Registry", transactions.size());

            int newCount = 0;
            for (PropertyTransaction tx : transactions) {
                if (propertyTransactionRepository.findByTransactionId(tx.getTransactionId()).isEmpty()) {
                    log.debug("Processing new transaction: {}", tx.getTransactionId());
                    // Link or create PropertyDetail
                    Optional<PropertyDetail> existingDetail = propertyDetailRepository.findByPostcodeAndAddress(tx.getPostcode(), tx.getAddress());
                    if (existingDetail.isPresent()) {
                        tx.setPropertyDetail(existingDetail.get());
                    } else {
                        log.debug("Fetching EPC details for {} - {}", tx.getPostcode(), tx.getAddress());
                        Optional<PropertyDetail> newDetail = epcService.fetchPropertyDetail(tx.getPostcode(), tx.getAddress());
                        if (newDetail.isPresent()) {
                            PropertyDetail savedDetail = propertyDetailRepository.save(newDetail.get());
                            tx.setPropertyDetail(savedDetail);
                        }
                        // Avoid rate limiting: sleep 200ms
                        Thread.sleep(200);
                    }
                    propertyTransactionRepository.save(tx);
                    newCount++;
                }
            }
            log.info("Poll complete. Saved {} new transactions.", newCount);
        } catch (Exception e) {
            log.error("Error during data poll: {}", e.getMessage(), e);
        }
    }
}
