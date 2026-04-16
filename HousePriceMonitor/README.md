# UK House Price Monitor

A Spring Boot application designed to monitor, enrich, and analyze UK house price data in specific geographic districts. The tool combines transaction data from HM Land Registry with physical property characteristics from the EPC register to provide granular market valuations.

## Overview
The application periodically polls the **HM Land Registry SPARQL API** for new property sales in three target districts (**Hartlepool TS27**, **Bideford EX39**, and **Kingston KT4**). It then enriches each transaction by querying the **DLUHC Energy Performance Certificate (EPC) API** to retrieve internal floor area and room counts, enabling calculations like Price per m² and Price per Room.

---

## Data Sources

### 1. HM Land Registry (Price Paid Data)
- **Interface:** SPARQL (Linked Data)
- **Data Retrieved:** Sale price, transaction date, property type (Detached, etc.), and full address.
- **Limitation:** Contains no information about the interior of the house (rooms, size).

### 2. DLUHC EPC Register
- **Interface:** REST API
- **Data Retrieved:** Total internal floor area (m²), number of habitable rooms, property age band, and built form.
- **Room Definition:** Uses the "Habitable Rooms" standard.
    - **Included:** Bedrooms, living rooms, studies, and large kitchen-diners.
    - **Excluded:** Bathrooms, toilets, hallways, garages, and small utility kitchens.

---

## Database Structure (PostgreSQL)

The application uses a local PostgreSQL database named `houseprices`.

### Tables:
1.  **`monitored_area`**: Configures the postcode districts to poll.
    - `id`, `postcode_district` (e.g., TS27), `name`.
2.  **`property_transaction`**: Stores the financial records.
    - `transaction_id`: Unique Land Registry ID.
    - `price`: Final sale price.
    - `transaction_date`, `postcode`, `address`, `property_type`.
    - `property_detail_id`: Foreign key to physical attributes.
3.  **`property_detail`**: Stores the physical attributes.
    - `total_floor_area`: Internal size in m².
    - `habitable_rooms`: Count of living/sleeping rooms.
    - `property_age_band`: Era of construction (e.g., "Pre-1900").
    - `built_form`: Refined structure (e.g., "Mid-Terrace").

---

## Business Logic & Calculations

### Similarity Engine
Properties are flagged as **"SIMILAR"** based on criteria defined in `HouseComparators.xml`.
- **Logic:** Must match `propertyType` and `ageBand` (if provided), and fall within the specified `minRooms`/`maxRooms` and `minArea`/`maxArea` ranges.

### Valuations
- **Price per m²:** `Sale Price / Total Floor Area`.
- **Price per Room:** `Sale Price / Habitable Rooms`.
- **Estimated Value (m²):** `Target actualArea (XML) × Avg Price per m² of Similar Houses`.
- **Estimated Value (Rooms):** `Target actualRooms (XML) × Avg Price per Room of Similar Houses`.

---

## Configuration (`HouseComparators.xml`)

This file allows you to define your search profile for each district.

### Key Elements & Attributes:
- **`actualArea`**: The size (m²) of the specific house style you are targeting.
- **`actualRooms`**: The number of habitable rooms (e.g., 5 for a 3-bed house) you are targeting.
- **`propertyType`**: Must match Land Registry values: `Detached`, `Semi-detached`, `Terraced`, `Flat/Maisonette`.
- **`ageBand`**: (Optional) Use strings like `Pre-1900`, `1930-1949`, `2007-2011`. Leave blank to ignore.

---

## Dashboard Elements

1.  **Market Comparison Chart:** A bar chart comparing the Average Price per m² for the whole market vs. your "Similar" house criteria across all districts.
2.  **District Comparison Cards:**
    *   **Market Avg:** The general average for all houses in that postcode.
    *   **Similar Avg:** The average for properties matching your XML criteria.
    *   **Estimated Values:** Two green valuation boxes providing a price range for your "target" property.
3.  **Recent Transactions (Tabs):**
    *   Sorted by **Similar** status first, then by **Price Descending**.
    *   Similar houses are highlighted in **blue**.
    *   Includes calculated columns for **Price/m²** and **Price/room**.

---

## Installation & Setup

1.  **Database:** Create a local Postgres database named `houseprices`.
2.  **Credentials:** Create `src/main/resources/application-local.properties`:
    ```properties
    spring.datasource.username=your_user
    spring.datasource.password=your_pass
    epc.api.email=your@email.com
    epc.api.key=your_epc_key
    ```
3.  **Run:** `./mvnw spring-boot:run`
4.  **Dashboard:** `http://localhost:8081/dashboard`
