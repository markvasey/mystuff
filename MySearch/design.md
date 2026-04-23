# MySearch - Design Specification

## 1. Architecture Overview
A Spring Boot 3.4.x application designed to aggregate personal data into a searchable database.
*   **Ingestion Engine**: Orchestrates the "Sync" process for Evernote, Dropbox, and Yahoo Mail.
*   **Storage Layer**: PostgreSQL with Full Text Search (FTS).
*   **Search UI**: A Thymeleaf-based web dashboard.

## 2. API Capabilities: "Since Datetime" Syncing
Syncing data incrementally requires different strategies for each service, as they don't all support exact "since datetime" queries:

*   **Evernote API**:
    *   **Capability**: Supports date filtering down to the day or exact UTC timestamp using the `NoteFilter` grammar (e.g., `updated:20260101T000000Z`).
    *   **Strategy**: We will store the `last_scan_at` timestamp and use it to query Evernote for newly updated notes.
*   **Dropbox API**:
    *   **Capability**: Does *not* support filtering `list_folder` by date.
    *   **Strategy**: Dropbox uses a **Cursor** system. We call `list_folder` to get an initial cursor, and store this `cursor` string instead of a datetime. Subsequent calls to `list_folder/continue` using this cursor will return only the files changed since that exact cursor was generated.
*   **Yahoo Mail (IMAP)**:
    *   **Capability**: IMAP's `SINCE` parameter only supports dates (e.g., `SINCE 1-Jan-2026`), not exact times.
    *   **Strategy**: We will use IMAP **UIDs (Unique Identifiers)**. UIDs strictly increase. We will save the highest `UID` seen in a folder and query for `UID <last_uid + 1>:*` on the next sync.

## 3. Database Schema (PostgreSQL)

### **Table: `search_items`**
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID / Serial | Primary Key (Internal) |
| `external_key` | TEXT | Unique ID from source (`guid` for Evernote, `id` for Dropbox, `uid` for Yahoo) |
| `source` | TEXT | `EVERNOTE`, `DROPBOX`, `YAHOO_MAIL` |
| `title` | TEXT | Note title, Filename, or Email Subject |
| `content` | TEXT | Full content of the item |
| `snippet` | TEXT | First 100 words (for quick display) |
| `item_date` | TIMESTAMP | Original creation/sent date |
| `scanned_at` | TIMESTAMP | Date of import |
| `search_vector` | TSVECTOR | Generated column for fast FTS indexing |

### **Table: `scan_metadata`**
| Column | Type | Description |
| :--- | :--- | :--- |
| `source` | TEXT | Primary Key (`EVERNOTE`, `DROPBOX`, `YAHOO_MAIL`) |
| `sync_token` | TEXT | The watermark for the next scan (Timestamp for Evernote, Cursor for Dropbox, UID for Yahoo) |

## 4. Ingestion Strategy
*   **Trigger**: A `/sync` POST endpoint triggered by a "Sync Now" button on the UI.
*   **Evernote**: Uses the Evernote SDK with OAuth access token. Fetches notes matching `updated:<sync_token>`, converts ENML (XML) to plain text.
*   **Dropbox**: Uses the Dropbox SDK with a Long-lived Access Token. Uses `list_folder/continue` with the saved `sync_token` (cursor). Filters by extensions (`txt, rtf, doc, docx, xls, xlsx, ppt, pptx, odt, pdf`). Uses Apache Tika to extract text.
*   **Yahoo Mail**: Uses Jakarta Mail (IMAP) with an App Password. Connects to "All Mail" or "Inbox", fetches messages matching UID range, and extracts text/plain or text/html (converted to text).

## 5. Search Implementation
*   **Query Engine**: PostgreSQL Full Text Search using `websearch_to_tsquery('english', :userInput)`. Supports quotes for phrases, `OR`, and `-` for negation.
*   **Ranking**: Results sorted by `ts_rank` (relevance) and then `item_date`.

## 6. User Interface
*   **Home**: A clean, centered search bar (Google style).
*   **Results**:
    *   List of cards showing `Title`, `Source`, `Original Date`, and a 20-word snippet.
    *   Clicking a result shows the full content.
    *   A "Sync" button in the header with a status indicator.

## 7. Technical Stack
*   **Backend**: Java 17+, Spring Boot 3.4.x, Spring Data JPA.
*   **Search**: PostgreSQL.
*   **Parsing**: Apache Tika (for PDF/Office).
*   **Frontend**: Thymeleaf.