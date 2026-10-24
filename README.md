# AtliQ T-Shirts: Database Q&A with LLM 👕

A natural language interface for querying a retail t-shirt inventory database. Ask questions in plain English and get answers powered by Google Gemini + LangChain — no SQL knowledge required.

---

## How It Works

```
User Question (Natural Language)
        ↓
Streamlit Frontend (frontend.py)
        ↓
LangChain Few-Shot Prompt Builder (langchain_helper.py)
  - Semantic similarity selects relevant examples from few_shots.py
  - Chroma vector store + Google Embeddings for example retrieval
        ↓
Google Gemini LLM generates SQL query
        ↓
SQLDatabaseChain executes query on MySQL (atliq_tshirts)
        ↓
LLM interprets result → Natural Language Answer
        ↓
Answer displayed in Streamlit UI
```

---

## Project Structure

```
RetailShop(LLM)/
├── frontend.py          # Streamlit UI
├── langchain_helper.py  # LangChain chain setup (LLM + DB + prompt)
├── few_shots.py         # Few-shot Q&A examples for prompt engineering
├── atliqdb.sql          # MySQL schema + seed data
├── .env                 # API keys and DB credentials (not committed)
├── .gitignore
└── main.ipynb           # Experimentation notebook
```

---

## Database Schema

Database: `atliq_tshirts`

### `t_shirts`
| Column | Type | Description |
|---|---|---|
| t_shirt_id | INT (PK) | Auto-increment primary key |
| brand | ENUM | `Van Huesen`, `Levi`, `Nike`, `Adidas` |
| color | ENUM | `Red`, `Blue`, `Black`, `White` |
| size | ENUM | `XS`, `S`, `M`, `L`, `XL` |
| price | INT | Price between $10–$50 |
| stock_quantity | INT | Units in stock |

### `discounts`
| Column | Type | Description |
|---|---|---|
| discount_id | INT (PK) | Auto-increment primary key |
| t_shirt_id | INT (FK) | References `t_shirts.t_shirt_id` |
| pct_discount | DECIMAL(5,2) | Discount percentage (0–100) |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini (`gemini-3.1-flash-lite`) |
| Embeddings | Google Generative AI Embeddings (`gemini-embedding-001`) |
| Orchestration | LangChain (`SQLDatabaseChain`, `FewShotPromptTemplate`) |
| Vector Store | ChromaDB (in-memory, for example selection) |
| Database | MySQL via `pymysql` |
| Frontend | Streamlit |
| Config | `python-dotenv` |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RetailShop(LLM)
```

### 2. Install dependencies

```bash
pip install streamlit langchain langchain-google-genai langchain-community langchain-experimental langchain-classic chromadb pymysql python-dotenv
```

### 3. Set up the MySQL database

```bash
mysql -u root -p < atliqdb.sql
```

This creates the `atliq_tshirts` database, tables, and populates them with sample data.

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

## Example Questions You Can Ask

| Question | What it does |
|---|---|
| "How many Nike t-shirts do we have in XS size and white color?" | Filters by brand, size, color |
| "What is the total inventory value for all S-size t-shirts?" | Aggregates price × stock |
| "How much revenue will we generate selling all Levi's shirts with discounts?" | JOIN with discounts table |
| "How many white Levi's shirts do we have?" | Simple filtered count |
| "What is the revenue from all large Nike shirts after discounts?" | Multi-table aggregation |

---

## Key Design: Few-Shot Prompting

The app uses **semantic similarity-based few-shot example selection** to improve SQL generation accuracy:

1. `few_shots.py` contains curated Question → SQLQuery → Answer examples.
2. These are embedded using Google Generative AI Embeddings and stored in a Chroma vector store.
3. At query time, the 2 most semantically similar examples are retrieved and injected into the prompt.
4. This guides the LLM to generate correct MySQL syntax for the specific schema.

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `GOOGLE_API_KEY` | Google AI Studio API key for Gemini LLM + Embeddings |
| `DB_USER` | MySQL username |
| `DB_PASSWORD` | MySQL password |
| `DB_HOST` | MySQL host (default: `localhost`) |
| `DB_NAME` | MySQL database name (default: `atliq_tshirts`) |

---

## Notes

- The `.env` file must never be committed to version control — it is listed in `.gitignore`.
- The Chroma vector store is created in-memory on every app run; no persistence is needed.
- `sample_rows_in_table_info=3` is passed to `SQLDatabase` so the LLM sees sample rows for better context.
