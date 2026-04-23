# MySearch - Personalised Google like Search across my personal data - Specification

## 1. Introduction
**Purpose:** Create a local search over my personal repositiores. Periodic data updates.
**Target Audience:** Me
**Core Objective:** Ability to search my local repositories quickly and easily like a local Google.

## 2. Functional Requirements
### 2.1 Data Retrieval
- The system shall fetch new data from Evernote, Dropbox, Yahoo mail on an ad hoc basis as requested by the user.
- **Source:** Evernote API (with Key), Yahoo Mail API (with OAuth), Dropbox API (with Key?)
- The system must authenticate with the API using either provided API key or OAuth.

### 2.2 Data Processing & Storage
- The system shall parse the emails, documents or notes, convert them to text or markdown (from xml or a proprietary format)
- **Storage Mechanism:** PostgreSQL database.
- Database: mysearch has been created
- Details in application-local.properties in this folder
- Single table in PostGres with columns:
 - Key - to be defined for each of EvernoteNote (is there a noteid?), YahooMail (is there an id, or else date/time?), DropboxFile (is there an id, else file path/name)
 - Source (EvernoteNote, YahooMail, DropboxFile)
 - Content (text of note, email or file - first 100 words).
- LastScanDateTime table with columns Source (EvernoteNote, YahooMail, DropboxFile) and LastScanDateTime, used to record when to next scan from

### 2.3 Search
- The postgres table should be searchable for various criteria eg ANDs, ORs, phrases, words
- Need to define Syntax - does Google Search have one?

### 2.5 User Interface
- **Dashboard:** A simple Web Dashboard like Google search with a single textbox.
- The UI should display found items as a list, with some metadata eg Source, Key, DateTime, first 20(?) words of text

## 3. Non-Functional Requirements
- **Language/Framework:** Java 17+, Spring Boot 3.4.x (per workspace standards).
- **Resilience:** The system should handle  API rate limits gracefully and implement retry logic for temporary failures.
- **Testing:** ALWAYS add a new test case to the existing test file (if one exists) or create a new test file to verify every new piece of functionality. Both exact and case-insensitive URL tests for the UI controllers.

## 4. Non-Fnctional Requirements
- I have created a application-local.properties in this folder with the postgres details
- I suggest any API keys go here
- If Yahoo Mail needs OAuth, maybe some private data needs to go here
