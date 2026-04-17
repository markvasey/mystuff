# Design: Sentiment, Completeness and Relevance Analysis

## 1. Overview
The analysis of political answers for sentiment, completeness, and relevance is a complex natural language task that is best handled by a Large Language Model (LLM). This design proposes using an LLM (e.g., GPT-4o, Claude 3.5, or a local Llama 3 instance via Ollama) to process each question-answer pair and return structured metrics.

## 2. Analysis Engine
### 2.1 AI Service Provider: Google Gemini
- **Framework:** Use **Spring AI** with the `spring-ai-google-ai-gemini-spring-boot-starter` (for Google AI Studio) or `spring-ai-vertex-ai-gemini-spring-boot-starter` (for Google Cloud Vertex AI).
- **Setup:** 
  - **API Key:** The system will require a `GOOGLE_AI_API_KEY` environment variable.
  - **Model:** `gemini-1.5-pro` (recommended for high-reasoning tasks like completeness analysis) or `gemini-1.5-flash` (faster and lower cost).

### 2.2 Structured Output (JSON Schema)
Google Gemini supports **JSON Mode** and **Schema Constraints**, ensuring the LLM always returns a valid JSON object that matches our database structure:
```json
{
  "sentiment": "Defensive",
  "tone": "Formal and slightly combative",
  "completeness": 65,
  "relevance": 80,
  "isDirectAnswer": false,
  "diversionTactics": ["Ad hominem", "Pivot to historical context"],
  "pointsAnswered": ["Action on NHS waiting lists"],
  "pointsMissed": ["Specific funding source for new staff"],
  "rational": "The speaker addressed the current state of the NHS but failed to answer the specific question regarding where the £2bn funding would be sourced from, instead blaming the previous administration."
}
```

## 3. Data Integration
### 3.1 Database Schema (PostgreSQL)
We will extend the `Utterance` or create a linked `AnalysisResult` table:
- `id` (UUID)
- `utterance_id` (FK to Utterance)
- `sentiment` (String)
- `completeness_score` (Integer 0-100)
- `relevance_score` (Integer 0-100)
- `is_direct_answer` (Boolean)
- `diversion_tactics` (JSONB/Text)
- `rational` (Text)
- `analyzed_at` (Timestamp)

### 3.2 Processing Flow
1. **Poll:** User clicks "Poll Now". `PMQsPoller` fetches new data.
2. **Ingest:** `PMQsService` saves new question-answer pairs to the DB.
3. **Queue Analysis:** For each *new* pair where Keir Starmer is the respondent, an `AnalysisService` is triggered.
4. **LLM Prompting:**
   - The system sends a prompt containing: *[Question Context] + [Answer Text] + [Detailed Instructions for scoring]*.
5. **Update:** The resulting metrics are saved to the `AnalysisResult` table.

## 4. UI Representation (Dashboard)
- **Score Badges:** Visual indicators (Green/Amber/Red) for Completeness and Relevance percentages.
- **Sentiment Labels:** Tags for "Honest", "Defensive", "Deflective", etc.
- **Detailed View:** A modal or expanded row that shows the "Rational/Reason" and lists specifically which points were missed or answered.
- **Aggregate Stats:** A sidebar showing "Average Completeness Score" for the current PMQs session.

## 5. Next Steps
1.  Confirm if you prefer a Cloud API (OpenAI/Anthropic) or a Local LLM (Ollama).
2.  If Cloud, please ensure API keys are available in your environment.
3.  If Local, I will configure the service to connect to `http://localhost:11434`.
