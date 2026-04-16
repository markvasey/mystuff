package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import com.example.housepricemonitor.model.MonitoredArea;
import com.example.housepricemonitor.model.PropertyDetail;
import com.example.housepricemonitor.model.PropertyTransaction;
import com.example.housepricemonitor.repository.MonitoredAreaRepository;
import com.example.housepricemonitor.repository.PropertyDetailRepository;
import com.example.housepricemonitor.repository.PropertyTransactionRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.time.LocalDate;
import java.util.Arrays;
import java.util.Optional;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

public class HousePricePollerTest {

    private HousePricePoller poller;
    private MonitoredAreaRepository monitoredAreaRepository;
    private PropertyTransactionRepository transactionRepository;
    private PropertyDetailRepository detailRepository;
    private LandRegistryService landRegistryService;
    private EpcService epcService;
    private ComparisonConfigService comparisonConfigService;

    @BeforeEach
    void setUp() {
        monitoredAreaRepository = mock(MonitoredAreaRepository.class);
        transactionRepository = mock(PropertyTransactionRepository.class);
        detailRepository = mock(PropertyDetailRepository.class);
        landRegistryService = mock(LandRegistryService.class);
        epcService = mock(EpcService.class);
        comparisonConfigService = mock(ComparisonConfigService.class);

        poller = new HousePricePoller(monitoredAreaRepository, transactionRepository, detailRepository, landRegistryService, epcService, comparisonConfigService);
    }

    @Test
    void testInitAreas() {
        ComparatorsConfig.DistrictCriteria criteria = new ComparatorsConfig.DistrictCriteria();
        criteria.setPostcode("KT4");
        criteria.setName("Kingston");
        
        when(comparisonConfigService.getAllCriteria()).thenReturn(Arrays.asList(criteria));
        when(monitoredAreaRepository.findByPostcodeDistrict("KT4")).thenReturn(Optional.empty());
        
        poller.initAreas();
        
        verify(monitoredAreaRepository, times(1)).save(any(MonitoredArea.class));
    }

    @Test
    void testPollNewData() {
        MonitoredArea area = new MonitoredArea();
        area.setPostcodeDistrict("KT4");
        when(monitoredAreaRepository.findAll()).thenReturn(Arrays.asList(area));

        PropertyTransaction tx = new PropertyTransaction();
        tx.setTransactionId("tx1");
        tx.setPostcode("KT4 1AA");
        tx.setAddress("1 Main St");
        when(landRegistryService.fetchTransactions(anyList(), any(LocalDate.class))).thenReturn(Arrays.asList(tx));

        when(transactionRepository.findByTransactionId("tx1")).thenReturn(Optional.empty());
        when(detailRepository.findByPostcodeAndAddress(anyString(), anyString())).thenReturn(Optional.empty());
        
        PropertyDetail detail = new PropertyDetail();
        when(epcService.fetchPropertyDetail(anyString(), anyString())).thenReturn(Optional.of(detail));
        when(detailRepository.save(any(PropertyDetail.class))).thenReturn(detail);

        poller.pollNewData();

        verify(transactionRepository).save(tx);
        verify(detailRepository).save(any(PropertyDetail.class));
        assert tx.getPropertyDetail() == detail;
    }
}
