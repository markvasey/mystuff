package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.ComparatorsConfig;
import com.example.housepricemonitor.model.PropertyTransaction;
import com.example.housepricemonitor.repository.PropertyTransactionRepository;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class HousePriceAnalyticsService {

    private final PropertyTransactionRepository transactionRepository;

    public HousePriceAnalyticsService(PropertyTransactionRepository transactionRepository) {
        this.transactionRepository = transactionRepository;
    }

    public Map<String, List<PropertyTransaction>> getTransactionsByDistrict() {
        return transactionRepository.findAll().stream()
                .collect(Collectors.groupingBy(tx -> tx.getPostcode().split(" ")[0]));
    }

    public List<PropertyTransaction> filterByCriteria(List<PropertyTransaction> transactions, ComparatorsConfig.DistrictCriteria criteria) {
        if (criteria == null) return transactions;
        return transactions.stream()
                .filter(tx -> isSimilar(tx, criteria))
                .collect(Collectors.toList());
    }

    public boolean isSimilar(PropertyTransaction tx, ComparatorsConfig.DistrictCriteria criteria) {
        if (criteria == null) return false;
        
        // Match Property Type
        if (criteria.getPropertyType() != null && !tx.getPropertyType().equalsIgnoreCase(criteria.getPropertyType())) {
            return false;
        }

        if (tx.getPropertyDetail() == null) return false;

        // Match Age Band (Style/Period) - Only if provided
        if (criteria.getAgeBand() != null && !criteria.getAgeBand().isBlank()) {
            if (!tx.getPropertyDetail().getPropertyAgeBand().equalsIgnoreCase(criteria.getAgeBand())) {
                return false;
            }
        }

        // Match Rooms
        if (criteria.getMinRooms() != null && tx.getPropertyDetail().getHabitableRooms() != null) {
            if (tx.getPropertyDetail().getHabitableRooms() < criteria.getMinRooms()) return false;
        }
        if (criteria.getMaxRooms() != null && tx.getPropertyDetail().getHabitableRooms() != null) {
            if (tx.getPropertyDetail().getHabitableRooms() > criteria.getMaxRooms()) return false;
        }

        // Match Area
        if (criteria.getMinArea() != null && tx.getPropertyDetail().getTotalFloorArea() != null) {
            if (tx.getPropertyDetail().getTotalFloorArea().doubleValue() < criteria.getMinArea()) return false;
        }
        if (criteria.getMaxArea() != null && tx.getPropertyDetail().getTotalFloorArea() != null) {
            if (tx.getPropertyDetail().getTotalFloorArea().doubleValue() > criteria.getMaxArea()) return false;
        }

        return true;
    }

    public BigDecimal calculateAveragePrice(List<PropertyTransaction> transactions) {
        if (transactions.isEmpty()) return BigDecimal.ZERO;
        BigDecimal sum = transactions.stream()
                .map(PropertyTransaction::getPrice)
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        return sum.divide(BigDecimal.valueOf(transactions.size()), 2, RoundingMode.HALF_UP);
    }

    public Map<String, BigDecimal> getAvgPriceByType(List<PropertyTransaction> transactions) {
        return transactions.stream()
                .collect(Collectors.groupingBy(PropertyTransaction::getPropertyType,
                        Collectors.collectingAndThen(
                                Collectors.toList(),
                                this::calculateAveragePrice
                        )));
    }

    public BigDecimal calculateAveragePricePerSqm(List<PropertyTransaction> transactions) {
        List<PropertyTransaction> withDetails = transactions.stream()
                .filter(tx -> tx.getPropertyDetail() != null && tx.getPropertyDetail().getTotalFloorArea() != null)
                .collect(Collectors.toList());
        
        if (withDetails.isEmpty()) return BigDecimal.ZERO;
        
        BigDecimal sumPricePerSqm = withDetails.stream()
                .filter(tx -> tx.getPropertyDetail().getTotalFloorArea().compareTo(BigDecimal.ZERO) > 0)
                .map(tx -> tx.getPrice().divide(tx.getPropertyDetail().getTotalFloorArea(), 2, RoundingMode.HALF_UP))
                .reduce(BigDecimal.ZERO, BigDecimal::add)
                .setScale(2, RoundingMode.HALF_UP);
        
        return sumPricePerSqm.divide(BigDecimal.valueOf(withDetails.size()), 2, RoundingMode.HALF_UP);
    }

    public BigDecimal calculateAveragePricePerRoom(List<PropertyTransaction> transactions) {
        List<PropertyTransaction> withRooms = transactions.stream()
                .filter(tx -> tx.getPropertyDetail() != null && tx.getPropertyDetail().getHabitableRooms() != null && tx.getPropertyDetail().getHabitableRooms() > 0)
                .collect(Collectors.toList());
        
        if (withRooms.isEmpty()) return BigDecimal.ZERO;
        
        BigDecimal sumPricePerRoom = withRooms.stream()
                .map(tx -> tx.getPrice().divide(BigDecimal.valueOf(tx.getPropertyDetail().getHabitableRooms()), 2, RoundingMode.HALF_UP))
                .reduce(BigDecimal.ZERO, BigDecimal::add);
        
        return sumPricePerRoom.divide(BigDecimal.valueOf(withRooms.size()), 2, RoundingMode.HALF_UP);
    }
}
