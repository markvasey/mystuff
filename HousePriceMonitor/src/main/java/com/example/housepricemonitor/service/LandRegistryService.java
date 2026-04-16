package com.example.housepricemonitor.service;

import com.example.housepricemonitor.dto.SparqlResponse;
import com.example.housepricemonitor.model.PropertyTransaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
public class LandRegistryService {

    private static final Logger log = LoggerFactory.getLogger(LandRegistryService.class);

    private final WebClient webClient;

    @Value("${landregistry.api.url}")
    private String apiUrl;

    public LandRegistryService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.build();
    }

    public List<PropertyTransaction> fetchTransactions(List<String> districts, LocalDate sinceDate) {
        String filterDistricts = districts.stream()
                .map(d -> "STRSTARTS(?postcode, \"" + d + "\")")
                .collect(Collectors.joining(" || "));

        String sparql = "PREFIX lrppi: <http://landregistry.data.gov.uk/def/ppi/>\n" +
                "PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>\n" +
                "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n" +
                "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" +
                "\n" +
                "SELECT ?trans ?price ?date ?postcode ?paon ?saon ?street ?locality ?town ?district ?county ?type\n" +
                "WHERE {\n" +
                "  ?trans lrppi:pricePaid ?price ;\n" +
                "         lrppi:transactionDate ?date ;\n" +
                "         lrppi:propertyAddress ?addr ;\n" +
                "         lrppi:propertyType ?type_url .\n" +
                "  ?addr lrcommon:postcode ?postcode ;\n" +
                "        lrcommon:paon ?paon .\n" +
                "  OPTIONAL { ?addr lrcommon:saon ?saon }\n" +
                "  OPTIONAL { ?addr lrcommon:street ?street }\n" +
                "  OPTIONAL { ?addr lrcommon:locality ?locality }\n" +
                "  OPTIONAL { ?addr lrcommon:town ?town }\n" +
                "  OPTIONAL { ?addr lrcommon:district ?district }\n" +
                "  OPTIONAL { ?addr lrcommon:county ?county }\n" +
                "  \n" +
                "  ?type_url skos:prefLabel ?type .\n" +
                "  \n" +
                "  FILTER (?date >= \"" + sinceDate + "\"^^xsd:date)\n" +
                "  FILTER (" + filterDistricts + ")\n" +
                "}\n" +
                "ORDER BY DESC(?date)\n" +
                "LIMIT 500";

        log.debug("Executing SPARQL query:\n{}", sparql);

        try {
            SparqlResponse response = webClient.post()
                    .uri(apiUrl)
                    .header("Content-Type", "application/sparql-query")
                    .header("Accept", "application/sparql-results+json")
                    .bodyValue(sparql)
                    .retrieve()
                    .bodyToMono(SparqlResponse.class)
                    .block();

            List<PropertyTransaction> transactions = new ArrayList<>();
            if (response != null && response.getResults() != null) {
                log.info("Received {} results from Land Registry", response.getResults().getBindings().size());
                for (Map<String, SparqlResponse.Binding> binding : response.getResults().getBindings()) {
                    PropertyTransaction tx = new PropertyTransaction();
                    tx.setTransactionId(binding.get("trans").getValue());
                    tx.setPrice(new BigDecimal(binding.get("price").getValue()));
                    tx.setTransactionDate(LocalDate.parse(binding.get("date").getValue()));
                    tx.setPostcode(binding.get("postcode").getValue());
                    
                    String paon = binding.get("paon").getValue();
                    String saon = binding.containsKey("saon") ? binding.get("saon").getValue() + ", " : "";
                    String street = binding.containsKey("street") ? binding.get("street").getValue() : "";
                    tx.setAddress(saon + paon + " " + street);
                    
                    tx.setPropertyType(binding.get("type").getValue());
                    transactions.add(tx);
                }
            } else {
                log.warn("Empty or null response from Land Registry SPARQL API");
            }
            return transactions;
        } catch (Exception e) {
            log.error("Error fetching transactions from Land Registry: {}", e.getMessage());
            throw e;
        }
    }
}
