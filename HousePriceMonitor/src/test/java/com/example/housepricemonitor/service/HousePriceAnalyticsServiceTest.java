package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import com.example.housepricemonitor.model.PropertyDetail;
import com.example.housepricemonitor.model.PropertyTransaction;
import com.example.housepricemonitor.repository.PropertyTransactionRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.when;

public class HousePriceAnalyticsServiceTest {

    private HousePriceAnalyticsService analyticsService;
    private PropertyTransactionRepository transactionRepository;

    @BeforeEach
    void setUp() {
        transactionRepository = Mockito.mock(PropertyTransactionRepository.class);
        analyticsService = new HousePriceAnalyticsService(transactionRepository);
    }

    @Test
    void testCalculateAveragePrice() {
        PropertyTransaction tx1 = new PropertyTransaction();
        tx1.setPrice(new BigDecimal("100000"));
        PropertyTransaction tx2 = new PropertyTransaction();
        tx2.setPrice(new BigDecimal("200000"));

        BigDecimal avg = analyticsService.calculateAveragePrice(Arrays.asList(tx1, tx2));
        assertEquals(new BigDecimal("150000.00"), avg);
    }

    @Test
    void testCalculateAveragePriceEmpty() {
        BigDecimal avg = analyticsService.calculateAveragePrice(Arrays.asList());
        assertEquals(BigDecimal.ZERO, avg);
    }

    @Test
    void testCalculateAveragePricePerSqm() {
        PropertyDetail detail1 = new PropertyDetail();
        detail1.setTotalFloorArea(new BigDecimal("100"));
        PropertyTransaction tx1 = new PropertyTransaction();
        tx1.setPrice(new BigDecimal("100000"));
        tx1.setPropertyDetail(detail1);

        PropertyDetail detail2 = new PropertyDetail();
        detail2.setTotalFloorArea(new BigDecimal("50"));
        PropertyTransaction tx2 = new PropertyTransaction();
        tx2.setPrice(new BigDecimal("100000"));
        tx2.setPropertyDetail(detail2);

        // (1000 + 2000) / 2 = 1500
        BigDecimal avg = analyticsService.calculateAveragePricePerSqm(Arrays.asList(tx1, tx2));
        assertEquals(new BigDecimal("1500.00"), avg);
    }

    @Test
    void testCalculateAveragePricePerRoom() {
        PropertyDetail detail1 = new PropertyDetail();
        detail1.setHabitableRooms(5);
        PropertyTransaction tx1 = new PropertyTransaction();
        tx1.setPrice(new BigDecimal("100000"));
        tx1.setPropertyDetail(detail1);

        PropertyDetail detail2 = new PropertyDetail();
        detail2.setHabitableRooms(4);
        PropertyTransaction tx2 = new PropertyTransaction();
        tx2.setPrice(new BigDecimal("100000"));
        tx2.setPropertyDetail(detail2);

        // (20000 + 25000) / 2 = 22500
        BigDecimal avg = analyticsService.calculateAveragePricePerRoom(Arrays.asList(tx1, tx2));
        assertEquals(new BigDecimal("22500.00"), avg);
    }

    @Test
    void testIsSimilar() {
        ComparatorsConfig.DistrictCriteria criteria = new ComparatorsConfig.DistrictCriteria();
        criteria.setPropertyType("Detached");
        criteria.setMinRooms(3);
        criteria.setMaxRooms(5);
        criteria.setMinArea(100.0);
        criteria.setMaxArea(200.0);

        PropertyTransaction tx = new PropertyTransaction();
        tx.setPropertyType("Detached");
        PropertyDetail detail = new PropertyDetail();
        detail.setHabitableRooms(4);
        detail.setTotalFloorArea(new BigDecimal("150"));
        tx.setPropertyDetail(detail);

        assertTrue(analyticsService.isSimilar(tx, criteria));

        // Test Type Mismatch
        tx.setPropertyType("Terraced");
        assertFalse(analyticsService.isSimilar(tx, criteria));
        tx.setPropertyType("Detached");

        // Test Room Mismatch
        detail.setHabitableRooms(2);
        assertFalse(analyticsService.isSimilar(tx, criteria));
        detail.setHabitableRooms(4);

        // Test Area Mismatch
        detail.setTotalFloorArea(new BigDecimal("250"));
        assertFalse(analyticsService.isSimilar(tx, criteria));
    }

    @Test
    void testIsSimilarOptionalAgeBand() {
        ComparatorsConfig.DistrictCriteria criteria = new ComparatorsConfig.DistrictCriteria();
        criteria.setPropertyType("Detached");
        criteria.setAgeBand("Pre-1900");

        PropertyTransaction tx = new PropertyTransaction();
        tx.setPropertyType("Detached");
        PropertyDetail detail = new PropertyDetail();
        detail.setPropertyAgeBand("1930-1949");
        tx.setPropertyDetail(detail);

        assertFalse(analyticsService.isSimilar(tx, criteria));

        criteria.setAgeBand(""); // Blank should match anything
        assertTrue(analyticsService.isSimilar(tx, criteria));

        criteria.setAgeBand(null); // Null should match anything
        assertTrue(analyticsService.isSimilar(tx, criteria));
    }
}
