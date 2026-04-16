package com.example.housepricemonitor.model;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

public class PropertyTransactionTest {

    @Test
    void testGetPricePerSqm() {
        PropertyTransaction tx = new PropertyTransaction();
        tx.setPrice(new BigDecimal("100000"));
        
        PropertyDetail detail = new PropertyDetail();
        detail.setTotalFloorArea(new BigDecimal("100"));
        tx.setPropertyDetail(detail);

        assertEquals(new BigDecimal("1000.00"), tx.getPricePerSqm());
    }

    @Test
    void testGetPricePerSqmNullDetail() {
        PropertyTransaction tx = new PropertyTransaction();
        tx.setPrice(new BigDecimal("100000"));
        assertNull(tx.getPricePerSqm());
    }

    @Test
    void testGetPricePerRoom() {
        PropertyTransaction tx = new PropertyTransaction();
        tx.setPrice(new BigDecimal("100000"));
        
        PropertyDetail detail = new PropertyDetail();
        detail.setHabitableRooms(5);
        tx.setPropertyDetail(detail);

        assertEquals(new BigDecimal("20000.00"), tx.getPricePerRoom());
    }
}
