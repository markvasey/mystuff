# PMQs Monitor & Analyser

The **PMQs Monitor & Analyser** is a Spring Boot application designed to track, store, and evaluate Prime Minister's Questions (PMQs) from the UK Parliament. It leverages the **TheyWorkForYou (TWFY)** API and **Google Gemini AI** to provide a searchable dashboard of political transcripts with automated sentiment and completeness analysis.

## 1. The Web Dashboard
The service provides a clean, responsive web interface (defaulting to `http://localhost:8083`) that allows users to:
*   **Navigate Sessions:** Use a dropdown to select specific Wednesday PMQs sessions stored in the database.
*   **Filter Content:** A "Hide Speaker" toggle allows users to remove procedural comments from the Speaker of the House, focusing solely on the debate between MPs and the Prime Minister.
*   **Trigger Scrapes:** A "Scrape 2026" button initiates a deep scan of the TWFY XML archives to find and ingest all PMQs sessions for the current year.
*   **On-Demand Analysis:** An "Analyze Session" button triggers the Gemini AI analysis for the currently viewed session, pairing questions with answers and generating metrics.

## 2. System Services

### 2.1 TWFYClient
The low-level API integration layer. It handles all communication with TheyWorkForYou.
*   **API Token:** Requires a TWFY API key (configured in `application-local.properties`).
*   **Endpoints:**
    *   `getXmlDirectoryIndex`: Scans the raw scraped XML folder on the TWFY servers.
    *   `getRawXmlFile`: Downloads specific XML transcripts.
    *   `getFullDebateByGid`: Uses a unique `GID` to retrieve a full, structured transcript of a specific debate section in JSON format.

### 2.2 HistoricalScraperService
The "Discovery" engine. It bypasses simple search and finds the official source records.
*   **XML Scanning:** It scans `https://www.theyworkforyou.com/pwdata/scrapedxml/debates/` for files matching Wednesday dates (e.g., `debates2026-03-25e.xml`).
*   **Regex Identification:** It uses a sophisticated regex to find the PMQs "block" within the XML:
    ```xml
    <major-heading ...> Prime Minister </major-heading>
    <speech id="uk.org.publicwhip/debate/2026-03-25e.290.1" ...>
    <minor-heading ...> Engagements </minor-heading>
    ```
*   **GID Extraction:** It extracts the `APIGid` (e.g., `2026-03-25e.290.1`) and hands it to the `PMQsService` for full ingestion.

### 2.3 PMQsService
The core business logic layer.
*   **Data Pairing:** Critically filters out `Party=Speaker` utterances. This allows the service to identify a question from an MP and the very next contribution as the Prime Minister's answer.
*   **Fragment Merging:** Automatically combines consecutive rows from the same speaker into a single, cohesive utterance using double line breaks. This ensures Gemini receives the full context of an answer.
*   **Database Management:** Handles the persistence and retrieval of `Utterance` objects.

### 2.4 AnalysisService
The AI reasoning layer using **Google Gemini**.
*   **Prompting:** Constructs a detailed prompt containing the MP's name/question and the Prime Minister's full merged answer.
*   **Schema Enforcement:** Instructs Gemini to return a specific JSON object.
*   **Rate Limiting:** Implements a 2-second delay between calls to comply with Gemini Free Tier limits.
*   **Transient-Aware:** Smart logic that allows AI analysis to run during tests without requiring a database connection.

## 3. Google Gemini Integration
The service is optimized for **gemini-flash-latest** via a direct REST call for maximum reliability.
*   **Setup:** You must obtain an API key from [Google AI Studio](https://aistudio.google.com/).
*   **Config:** Set `spring.ai.google.genai.api-key` in your `application-local.properties`.
*   **Model Reasoning:** Gemini analyzes the PM's answer for **Sentiment** (Defensive, Honest, etc.), **Completeness** (0-100%), and **Relevance**. It also explicitly lists "Points Missed" if the PM dodged a specific part of a question.

## 4. Data Examples

### 4.1 Debate Transcript JSON (TWFY)
```json
[
  {
    "gid": "2026-03-25e.290.3",
    "hdate": "2026-03-25",
    "speaker": {
      "name": "Cat Smith",
      "party": "Labour",
      "person_id": "25432"
    },
    "body": "<p>If he will list his official engagements...</p>"
  },
  {
    "gid": "2026-03-25e.290.4",
    "speaker": {
      "name": "Keir Starmer",
      "person_id": "25353"
    },
    "body": "<p>An attack on Britain's Jewish community...</p>"
  }
]
```

### 4.2 Analysis Result JSON (Gemini)
```json
{
  "sentiment": "Combative",
  "tone": "Formal and defensive",
  "completeness": 40,
  "relevance": 95,
  "isDirectAnswer": false,
  "diversionTactics": ["Pivoting to previous government", "Attacking the questioner"],
  "pointsAnswered": ["Economic stability"],
  "pointsMissed": ["Specific timeline for sewage reduction"],
  "rational": "The Prime Minister correctly identified the economic impact but failed to provide the requested date for the sewage infrastructure update, instead blaming the previous administration's lack of funding."
}
```

## 5. Helpful Notes
*   **Wednesdays Only:** The dashboard automatically filters the database to only show contributions from Wednesdays.
*   **APIGids:** The GID format is `YYYY-MM-DD[letter].[num1].[num2]`. Incrementing `num2` is a key utility for navigating the transcript stream.
*   **Speaker Filtering:** Hiding the Speaker is enabled by default to make the "Analyze Session" pairings accurate.
