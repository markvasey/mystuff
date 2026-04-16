package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.EpcResponse;
import com.example.housepricemonitor.model.PropertyDetail;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class EpcService {

    private static final Logger log = LoggerFactory.getLogger(EpcService.class);

    private final WebClient webClient;

    @Value("${epc.api.email}")
    private String apiEmail;

    @Value("${epc.api.key}")
    private String apiKey;

    @Value("${epc.api.base-url}")
    private String baseUrl;

    public EpcService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.build();
    }

    public Optional<PropertyDetail> fetchPropertyDetail(String postcode, String address) {
        // EPC API expects postcode without spaces or with specific format
        String cleanPostcode = postcode.replace(" ", "");
        
        // Authorization: Basic base64(email:api-key)
        String authString = apiEmail + ":" + apiKey;
        String authHeader = "Basic " + Base64.getEncoder().encodeToString(authString.getBytes(StandardCharsets.UTF_8));

        log.debug("Querying EPC API for postcode: {} at {}", cleanPostcode, baseUrl);

        try {
            EpcResponse response = webClient.get()
                    .uri(baseUrl + "/search?postcode=" + cleanPostcode)
                    .header("Authorization", authHeader)
                    .header("Accept", "application/json")
                    .retrieve()
                    .bodyToMono(EpcResponse.class)
                    .block();

            if (response != null && response.getRows() != null && !response.getRows().isEmpty()) {
                // Fuzzy match address
                for (Map<String, Object> row : response.getRows()) {
                    String epcAddress = (String) row.get("address");
                    if (isMatch(address, epcAddress)) {
                        PropertyDetail detail = new PropertyDetail();
                        detail.setAddress(epcAddress);
                        detail.setPostcode(postcode);
                        
                        Object area = row.get("total-floor-area");
                        if (area != null) detail.setTotalFloorArea(new BigDecimal(area.toString()));
                        
                        Object rooms = row.get("number-habitable-rooms");
                        if (rooms != null) detail.setHabitableRooms(Integer.parseInt(rooms.toString()));
                        
                        detail.setPropertyAgeBand((String) row.get("property-age-band"));
                        detail.setBuiltForm((String) row.get("built-form"));
                        return Optional.of(detail);
                    }
                }
            }
        } catch (Exception e) {
            log.error("Error querying EPC API: {}", e.getMessage());
        }
        return Optional.empty();
    }

    private boolean isMatch(String addr1, String addr2) {
        if (addr1 == null || addr2 == null) return false;
        // Simple fuzzy match: check if the first part of the address (house number) matches
        String house1 = addr1.split(" ")[0].toLowerCase();
        String house2 = addr2.split(" ")[0].toLowerCase();
        return house1.equals(house2);
    }
}
