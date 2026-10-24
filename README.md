# AtliQ T-Shirts: Database Q&A with LLM 👕

A natural language interface for querying a retail t-shirt inventory database. Ask questions in plain English and get SQL-powered answers — no SQL knowledge required. Built with Google Gemini, LangChain, ChromaDB, MySQL, and Streamlit.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Project Structure](#project-structure)
4. [Database Schema](#database-schema)
5. [Tech Stack](#tech-stack)
6. [Core Components](#core-components)
   - [frontend.py](#frontendpy)
   - [langchain_helper.py](#langchain_helperpy)
   - [few_shots.py](#few_shotspy)
   - [atliqdb.sql](#atliqdbsql)
7. [Key Design: Few-Shot Prompting](#key-design-few-shot-prompting)
8. [LLM Prompt Engineering](#llm-prompt-engineering)
9. [Setup & Installation](#setup--installation)
10. [Environment Variables](#environment-variables)
11. [Example Questions](#example-questions)
12. [Troubleshooting](#troubleshooting)
13. [Notes & Limitations](#notes--limitations)

---

## Overview

AtliQ T-Shirts is an AI-powered Q&A system that lets non-technical users query a MySQL inventory database using plain English. Under the hood:

- The user's question is matched against curated SQL examples using **semantic similarity** (ChromaDB + Google Embeddings).
- The most relevant examples are injected into a **few-shot prompt** sent to **Google Gemini**.
- Gemini generates a MySQL query, which is executed against the live database via **LangChain's SQLDatabaseChain**.
- The raw SQL result is interpreted by the LLM and returned as a human-readable answer.

---

## Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     User (Browser)                          │
│              Asks a natural language question               │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                        │
│                     (frontend.py)                           │
│         st.text_input → calls get_few_shot_db_chain()       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              LangChain Helper (langchain_helper.py)         │
│                                                             │
│  1. Embeds user question using GoogleGenerativeAIEmbeddings │
│     (model: gemini-embedding-001)                           │
│                                                             │
│  2. SemanticSimilarityExampleSelector queries ChromaDB      │
│     → retrieves top-2 most similar few-shot examples        │
│     from few_shots.py                                       │
│                                                             │
│  3. FewShotPromptTemplate builds the final prompt:          │
│     [MySQL system instructions]                             │
│     + [2 similar Q/SQL/Result/Answer examples]              │
│     + [DB table schema + 3 sample rows]                     │
│     + [User question]                                       │
│                                                             │
│  4. GoogleGenerativeAI (gemini-3.1-flash-lite, temp=0.3)    │
│     generates a MySQL query                                 │
│                                                             │
│  5. SQLDatabaseChain executes query on MySQL                │
│     → returns raw result                                    │
│                                                             │
│  6. LLM interprets raw result → natural language answer     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   MySQL Database                            │
│               (atliq_tshirts via pymysql)                   │
│          Tables: t_shirts, discounts                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Answer displayed in Streamlit UI               │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RetailShop(LLM)/
├── frontend.py          # Streamlit UI — text input + answer display
├── langchain_helper.py  # Core logic: LLM, DB, embeddings, chain setup
├── few_shots.py         # Curated Q→SQL→Result→Answer examples
├── atliqdb.sql          # MySQL schema, stored procedure, seed data
├── .env                 # Secret keys and DB credentials (never commit)
├── .gitignore           # Excludes .env and other sensitive files
└── main.ipynb           # Jupyter notebook for experimentation
```

---

## Database Schema

Database: `atliq_tshirts`

### `t_shirts`

| Column | Type | Constraints | Description |
|---|---|---|---|
| t_shirt_id | INT | PK, AUTO_INCREMENT | Unique identifier |
| brand | ENUM | NOT NULL | `Van Huesen`, `Levi`, `Nike`, `Adidas` |
| color | ENUM | NOT NULL | `Red`, `Blue`, `Black`, `White` |
| size | ENUM | NOT NULL | `XS`, `S`, `M`, `L`, `XL` |
| price | INT | CHECK (10–50) | Price in USD |
| stock_quantity | INT | NOT NULL | Units currently in stock |

- Unique constraint on `(brand, color, size)` — no duplicate combinations.
- Populated via a stored procedure `PopulateTShirts()` that inserts up to 100 random records.

### `discounts`

| Column | Type | Constraints | Description |
|---|---|---|---|
| discount_id | INT | PK, AUTO_INCREMENT | Unique identifier |
| t_shirt_id | INT | FK → t_shirts | Which t-shirt gets the discount |
| pct_discount | DECIMAL(5,2) | CHECK (0–100) | Discount percentage |

- 10 discount records are seeded for t_shirt_id 1–10.
- Discounts are applied using: `price × stock × ((100 - pct_discount) / 100)`

### Entity Relationship

```
t_shirts (t_shirt_id) ──< discounts (t_shirt_id)
         [1]                      [0..*]
```

---

## Tech Stack

| Component | Technology | Version / Model |
|---|---|---|
| LLM | Google Gemini | `gemini-3.1-flash-lite` |
| Embeddings | Google Generative AI | `gemini-embedding-001` |
| LLM Orchestration | LangChain | `SQLDatabaseChain`, `FewShotPromptTemplate` |
| Example Selection | LangChain | `SemanticSimilarityExampleSelector` |
| Vector Store | ChromaDB | In-memory |
| Database | MySQL | via `pymysql` driver |
| Frontend | Streamlit | — |
| Config | python-dotenv | `.env` file |

---

## Core Components

### `frontend.py`

The Streamlit UI. Minimal by design — just a text input and an answer display.

```python
question = st.text_input("Question: ")
if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)
    st.header("Answer")
    st.write(response)
```

- A new chain (including a fresh Chroma vector store) is created on every question submission.
- No session state or caching is used — each run is stateless.

---

### `langchain_helper.py`

The core of the application. The `get_few_shot_db_chain()` function:

1. **Connects to MySQL** using `SQLDatabase.from_uri()` with `pymysql`. Passes `sample_rows_in_table_info=3` so the LLM receives 3 sample rows per table for better schema understanding.

2. **Initializes Gemini LLM** (`gemini-3.1-flash-lite`) with `temperature=0.3` — low temperature for deterministic, accurate SQL generation.

3. **Creates embeddings** using `GoogleGenerativeAIEmbeddings` with `gemini-embedding-001`.

4. **Builds a Chroma vector store** from the few-shot examples in `few_shots.py`. Each example's fields are joined into a single string and embedded.

5. **Sets up `SemanticSimilarityExampleSelector`** with `k=2` — retrieves the 2 most relevant examples for each user question.

6. **Constructs `FewShotPromptTemplate`** combining:
   - A MySQL expert system prompt (`mysql_prompt`)
   - The 2 dynamically selected examples
   - LangChain's built-in `PROMPT_SUFFIX` (appends table info and the user question)

7. **Returns `SQLDatabaseChain`** with `verbose=True` — logs the generated SQL and result to the console.

---

### `few_shots.py`

Contains 6 hand-crafted examples that cover the main query patterns in this database:

| # | Pattern | Description |
|---|---|---|
| 1 | Filtered COUNT | Stock for a specific brand + color + size |
| 2 | Aggregation | Total inventory value for a size |
| 3 | JOIN + discount | Revenue after applying discounts (Levi's) |
| 4 | Simple aggregation | Revenue without discount (Levi's) |
| 5 | Filtered SUM | Stock count for brand + color |
| 6 | JOIN + discount + filter | Revenue after discounts for brand + size (Nike L) |

Each example has 4 fields: `Question`, `SQLQuery`, `SQLResult`, `Answer`.

The discount revenue formula used in examples 3 and 6:
```sql
SUM(total_amount * ((100 - COALESCE(pct_discount, 0)) / 100))
```
`COALESCE` handles t-shirts with no discount entry (treats as 0% discount).

---

### `atliqdb.sql`

Sets up the entire database from scratch:

1. Creates `atliq_tshirts` database and both tables with constraints.
2. Defines stored procedure `PopulateTShirts()` that loops up to 100 times, generating random brand/color/size/price/stock combinations. Duplicate `(brand, color, size)` combinations are silently skipped via a `CONTINUE HANDLER FOR 1062`.
3. Calls `PopulateTShirts()` to seed the `t_shirts` table.
4. Inserts 10 discount records for t_shirt_id 1–10 with discounts ranging from 5% to 45%.

---

## Key Design: Few-Shot Prompting

Standard zero-shot prompting often produces incorrect SQL for domain-specific schemas. This project solves that with **semantic similarity-based few-shot example selection**:

```
few_shots.py examples
        │
        ▼ embed with gemini-embedding-001
ChromaDB vector store (in-memory)
        │
        ▼ cosine similarity search (k=2) against user question
2 most relevant examples
        │
        ▼ injected into FewShotPromptTemplate
Final prompt sent to Gemini
```

**Why this works better than static few-shot:**
- A fixed set of examples in the prompt wastes tokens on irrelevant examples.
- Dynamic selection ensures the LLM always sees the most relevant SQL patterns for the current question.
- With only 6 examples and k=2, the prompt stays concise while covering the key query types.

---

## LLM Prompt Engineering

The final prompt sent to Gemini has this structure:

```
[MySQL Expert System Instructions]
  - Generate syntactically correct MySQL
  - Use LIMIT {top_k} unless user specifies
  - Only query needed columns, wrap in backticks
  - Use CURDATE() for date-related questions
  - Output format: Question / SQLQuery / SQLResult / Answer

[Example 1 — most similar to user question]
Question: ...
SQLQuery: ...
SQLResult: ...
Answer: ...

[Example 2 — second most similar]
Question: ...
SQLQuery: ...
SQLResult: ...
Answer: ...

[Table schema with 3 sample rows per table]

Question: <user's actual question>
SQLQuery:
```

The LLM completes from `SQLQuery:` onward, generating the query, which `SQLDatabaseChain` then executes.

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- MySQL Server running locally
- Google AI Studio API key

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd "RetailShop(LLM)"
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-google-genai langchain-community langchain-experimental langchain-classic chromadb pymysql python-dotenv
```

### 3. Set up the MySQL database

```bash
mysql -u root -p < atliqdb.sql
```

This will:
- Create the `atliq_tshirts` database
- Create `t_shirts` and `discounts` tables
- Populate `t_shirts` with up to 100 random records
- Insert 10 discount records

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=<your_google_api_key>
DB_USER=<your_mysql_username>
DB_PASSWORD=<your_mysql_password>
DB_HOST=localhost
DB_NAME=atliq_tshirts
```

Get your Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 5. Run the app

```bash
streamlit run frontend.py
```

Open your browser at `http://localhost:8501`.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | Yes | — | Google AI Studio API key (Gemini LLM + Embeddings) |
| `DB_USER` | Yes | — | MySQL username |
| `DB_PASSWORD` | Yes | — | MySQL password |
| `DB_HOST` | Yes | `localhost` | MySQL host address |
| `DB_NAME` | Yes | `atliq_tshirts` | MySQL database name |

---

## Example Questions

| Question | SQL Pattern Used |
|---|---|
| "How many Nike t-shirts do we have in XS size and white color?" | `SUM(stock_quantity)` with `WHERE brand + color + size` |
| "What is the total inventory value for all S-size t-shirts?" | `SUM(price * stock_quantity)` with `WHERE size` |
| "How much revenue will we generate selling all Levi's shirts with discounts?" | `LEFT JOIN discounts` + `COALESCE(pct_discount, 0)` |
| "How much revenue without discount for all Levi's shirts?" | `SUM(price * stock_quantity)` with `WHERE brand` |
| "How many white Levi's shirts do we have?" | `SUM(stock_quantity)` with `WHERE brand + color` |
| "Revenue from all large Nike shirts after discounts?" | `LEFT JOIN discounts` + `WHERE brand + size` |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'langchain_classic'`**
```bash
pip install langchain-classic
```

**`Access denied for user` (MySQL error)**
- Verify `DB_USER` and `DB_PASSWORD` in `.env` are correct.
- Ensure the MySQL user has SELECT privileges on `atliq_tshirts`.

**`google.api_core.exceptions.InvalidArgument` (Gemini API error)**
- Check that `GOOGLE_API_KEY` in `.env` is valid and not expired.
- Verify the model name `models/gemini-3.1-flash-lite` is available in your region.

**Empty or wrong SQL generated**
- Add more relevant examples to `few_shots.py` that match the query pattern.
- Increase `k` in `SemanticSimilarityExampleSelector` from 2 to 3.

**Chroma `InvalidDimensionException`**
- This can occur if the embedding model changes between runs. Since the store is in-memory, it resets on every run — restart the app.

---

## Notes & Limitations

- The Chroma vector store is **in-memory only** — rebuilt on every app run. No disk persistence is needed or used.
- `sample_rows_in_table_info=3` sends 3 sample rows per table to the LLM for schema context. Increase this if the LLM struggles with data types.
- `temperature=0.3` is intentionally low to reduce hallucinated SQL. Set to `0` for fully deterministic output.
- `verbose=True` on `SQLDatabaseChain` prints the generated SQL and result to the terminal — useful for debugging.
- The `.env` file must **never** be committed to version control. Confirm it is listed in `.gitignore`.
- The app does **not** validate or sanitize user input before passing it to the LLM. For production use, add input validation.
- Only `SELECT` queries are expected. The LLM prompt does not explicitly block `INSERT`/`UPDATE`/`DELETE` — consider adding guardrails for production deployments.
