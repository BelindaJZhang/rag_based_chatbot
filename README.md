## Project Summary

This repository contains a fully working **RAG-based chatbot** for an e-commerce setting (“Fashion Forward Hub”).
It is an adaptation and significant extension of a Coursera assignment, redesigned to run **end-to-end locally** using:

### 🔹 Google Gemini API

Used for:

- text generation
- task routing
- JSON metadata extraction
- high-quality embeddings (`text-embedding-004`)

### 🔹 Local Weaviate (Docker)

I deployed my own Weaviate instance via Docker and:

- created the schema programmatically
- embedded and ingested all product data
- implemented hybrid search and reranking
- handled GRPC + REST connections locally

### 🔹 Core Engineering Features

This project demonstrates:

- LLM routing to separate FAQ vs Product queries
- conditional parameter setting (temperature, top_p, model)
- structured JSON extraction & parsing for metadata filters
- semantic + keyword retrieval via hybrid search
- reranking via Weaviate’s ranking module
- fully local RAG loop with no cloud dependencies except Gemini API
- a notebook chat UI for interactive testing

Together, these components form a **complete RAG system** that is deployable, explainable, and production-inspired — far beyond a standard course assignment.

## Project Architecture

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline using:

- **Google Gemini API** – for text generation and embeddings (`text-embedding-004`)
- **Weaviate (Docker)** – running locally as the vector database
- **LLM Routing** – dynamically classify queries as FAQ or Product
- **JSON-Based Metadata Extraction** – convert natural language queries into structured filters
- **Hybrid Search + Reranking** – combine keyword + semantic similarity for better retrieval
- **ChatBot Orchestration** – unify routing, metadata parsing, retrieval, and Gemini response generation
- **Notebook UI** – simple chat interface inside Jupyter

### 🔄 End-to-end Flow

1. **User query** enters the chatbot
2. The **LLM router** classifies it → *FAQ* or *Product*
3. For Product queries:
   - Gemini produces **JSON metadata** (gender, category, price, color, season…)
   - Notebook parses JSON → builds Weaviate filters
4. The system performs **Hybrid Search** in Weaviate
5. Weaviate returns **top product hits**
6. The final Gemini call generates a **grounded answer** that cites retrieved items
7. The chat UI displays results

```mermaid
flowchart TD
    U["User / ChatWidget"]
        --> ROUTE["answer_query()"]

    %% Routing
    ROUTE -->|FAQ| FAQ["FAQ Flow"]
    ROUTE -->|Product| PROD["Product Flow"]
    ROUTE -->|Other| FALLBACK["Fallback LLM Answer"]

    %% FAQ
    FAQ --> LLM_FAQ["LLM Answer (FAQ)"]
    LLM_FAQ --> U

    %% Product RAG
    PROD --> META["Extract Metadata"]
    META --> FILTERS["Build Filters"]
    PROD --> EMBED["Embed Query"]

    FILTERS --> RETRIEVE["Retrieve from Weaviate"]
    EMBED --> RETRIEVE
    RETRIEVE --> CONTEXT["Build Product Context"]

    CONTEXT --> PROMPT["Craft Final Prompt"]
    PROMPT --> LLM_PROD["LLM Answer (Products)"]
    LLM_PROD --> U

    %% Fallback
    FALLBACK --> U

```

## Features / Capabilities

**✔ Product RAG**

- Vector-based search using text-embedding-004; Metadata filtering using LLM-generated attributes: gender, masterCategory, articleType, baseColour, usage, season, price
- Retrieves top-N similar items from Weaviate
- Generates product context summaries

**✔ FAQ Answering**

- Lightweight
- deterministic responses
- uses structured FAQ layout,
- LLM invoked only as needed

**✔ Intelligent Routing:**

- FAQ vs Product vs Other handled automatically
- Custom classification step for creative vs technical product tasks
- Tailored prompts and LLM parameters per task type

**✔ Scalable Product Ingestion**

- ~44,424 fashion products,
- Full cleaning pipeline, Schema validationm, Duplicate detection, Error-handled batch insertion into Weaviate

**✔ Clean Utility Modules**

- generate_metadata()
- embed_query()
- generate_items_context()
- get_filter_map_by_metadata()
- decide_task_nature()
- get_task_parameters()
- Unified generate_with_single_input() and generate_with_multiple_input()

## Dataset Description

### Product Dataset

* **Size:** ~44k products
* **Source:** fashion e-commerce catalogue (provided in the course)
* **Fields used:**
  * title
  * gender
  * masterCategory
  * articleType
  * baseColour
  * usage
  * season
  * price

### 🔹 Cleaning & Validation

* Removed incomplete rows
* Standardized categories
* Enforced unique `product_id`
* Ensured stable schema for Weaviate collection

### 🔹 FAQ Dataset

* Small structured table of common customer questions
* Stored as DataFrame / JSON
* Loaded and embedded directly in notebook

## Project Structure

```text
RAG_BASED_CHATBOT
├── .gitignore
├── docker-compose.yml
├── environment.yml
├── README.md
│
├── dataset/
│   ├── clothes.csv
│   ├── clothes_json.joblib
│   └── faq.joblib
│
├── images/
│   └── toc.png
│
├── notebooks/
│   ├── products_collection.ipynb
│   └── RAG_based_chatbot_for_Fasion_Forward_Hub.ipynb
│
├── src/
    └── utils.py
```

## Example Usage / Demo

Below are some sample interactions with the Fashion Forward Hub RAG chatbot, demonstrating how the system handles different types of user queries.

### 1. 👔 Creative Styling / Recommendation Queries

The chatbot can generate outfit ideas by retrieving relevant products and combining them into coherent suggestions.

![Creative styling example](images/demo_styling.png)

---

### 2. ❓ FAQ & Policy Questions

For store policies (returns, delivery, exchanges), the chatbot uses the FAQ collection for accurate, grounded responses.

![FAQ example](images/demo_faq.png)

---

### 3. 👕 Product Search & Technical Queries

Users can search the catalog (e.g., “blue T-shirts for men”), and the system retrieves matching items using vector + BM25 hybrid search.

![Product query example](images/demo_product_search.png)

## Limitations

- Notebook-based prototype (not a deployable app)
- Retrieval relies only on vector + metadata filtering
- No BM25 hybrid search
- No re-ranking applied yet
- Not optimized for long-form documents (no chunking stage)
- Creative styling answers may vary due to LLM creativity

## Future Work

- Add Weaviate Rerank module for improved relevance
- Implement BM25 + vector hybrid search
- Add product image retrieval
- Deploy system via Streamlit / FastAPI
- Add memory & conversation history
- Introduce traceability (LangSmith, OpenTelemetry)
- Build re-usable Python modules from notebook code

## Acknowledgments

- This project is adapted from: Coursera – “Retrieval Augmented Generation (RAG)”
- The core course concepts inspired the architecture, but ingestion pipeline, metadata design, retrieval logic, prompts, and structural enhancements were custom-developed.

## 📬 Contact

If you're interested in AI engineering, RAG systems, or production LLM pipelines — feel free to reach out!

Juan Zhang: https://www.linkedin.com/in/juan-zhang-finance-professional/
