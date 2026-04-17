# Hansard Prime Minister's Questions (PMQs) Monitor & Analyser - Specification

## 1. Introduction
**Purpose:** To on-demand poll, retrieve, and store Prime Minister's Questions (PMQs) from the UK Parliament, with a specific focus on answers given by Keir Starmer, and their sentiment, completeness and relevance.
**Target Audience:** People interested in political honesty and integrity.
**Core Objective:** Provide an automated way to track and analyse PMQs transcripts via a web dashboard, utilizing community-driven API data, specifically anlaysing answers for their sentiment, completeness and relevance.

## 2. Functional Requirements
### 2.1 Data Retrieval
- The system shall fetch new PMQs data on an ad hoc basis as requested by the user.
- **Source:** TheyWorkForYou API (using `getDebates` or similar endpoint for Commons PMQs).
- The system must authenticate with the TheyWorkForYou API using a provided API key.

### 2.2 Data Processing & Storage
- The system shall parse the raw JSON API responses into structured domain models (e.g., DebateSession, Utterance, Speaker).
- **Storage Mechanism:** PostgreSQL database.
- The system must prevent duplicate entries for the same PMQs session or individual utterance by using unique identifiers provided by the API.

### 2.3 Search & Flagging
- **Specific Monitoring:** The system will specifically focus on questions to "Keir Starmer" and his replies.
- The system should focus on prime ministers questions.
- The system should record the question and the answer given by Kier Starmer.
- Where another minister is representing Kier Starmer, these conversations should be processed but flagged as "On behalf of Kier Starmer".

### 2.4 Sentiment, Completeness and Relevance Analysis
- The primary purpose of the program is to analyse the answers given to the questions.
- The aspects to analyse are:
 - 1. Sentiment - what was the sentiment, mood, tone of the answer - was it "firm", "open", "honest", "forceful", "aggressive", "concilliatory", "defensive" etc
 - 2. Completeness - did the answer cover all the aspects of the question, what percentage of the points asked where answered?
 - 3. Relevance - was the answer relevant to the question? Was the question answered? Was the subject changed? Was the answer a question istelf? Where diversionary tactics used? Was a answer promised in the future?
 - 4. Rational/Reason - provide evidence for the completeness and relevance metrics

### 2.5 User Interface
- **Dashboard:** A simple Web Dashboard (likely using Thymeleaf, based on workspace patterns) to view ingested PMQs.
- The UI should display the latest PMQs sessions.
- The UI should have a specific section or visual indicator for the tracked MP (Keir Starmer).
- The UI should present statistics around the Sentiment, Completeness and Relevance Analysis

## 3. Non-Functional Requirements
- **Language/Framework:** Java 17+, Spring Boot 3.4.x (per workspace standards).
- **Resilience:** The system should handle TheyWorkForYou API rate limits gracefully and implement retry logic for temporary failures.
- **Testing:** ALWAYS add a new test case to the existing test file (if one exists) or create a new test file to verify every new piece of functionality. Both exact and case-insensitive URL tests for the UI controllers.

## 4. Proposed Architecture Overview
1.  **Poller Service (`PMQsPoller`):** A Spring service that runs on-demand ("Poll Now" button in UI) to query TheyWorkForYou API for new PMQs.
2.  **Service Layer (`PMQsService`):** Handles the business logic of parsing API responses, checking for existing records in the database, and saving new utterances.
3.  **Repository Layer (`PMQsRepository`):** Spring Data JPA repositories interfacing with PostgreSQL.
4.  **Web Controller (`DashboardController`):** Serves the Thymeleaf dashboard templates and provides data to the view.
