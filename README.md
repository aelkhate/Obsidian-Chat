# Obsidian Intent-Aware GenAI Assistant

## Overview

This repository implements a **local, intent-aware, multi-agent Generative AI assistant** for querying and reasoning over a personal Obsidian knowledge vault.

The system is designed to provide **grounded, citation-backed answers**, avoid hallucinations, and correctly handle different user query intents such as factual question answering, full document extraction, and vault browsing. It is fully local and does not rely on proprietary APIs.

The project is built to satisfy Master’s-level course requirements in Generative AI and LLM Systems, including Retrieval-Augmented Generation (RAG), tool/function calling, multi-agent workflows, fine-tuning via PEFT, evaluation, and safety considerations.

---

## Key Features

- Fully local execution (no OpenAI or external APIs)
- Retrieval-Augmented Generation using FAISS
- Intent-aware routing (QA vs document extraction vs browsing)
- Multi-agent architecture with clear role separation
- Deterministic tools to prevent hallucinations
- Explicit source citations or refusal
- Explicit domain ontology (PARA methodology)
- Behavioral fine-tuning strategy (LoRA / PEFT)
- Reproducible indexing and evaluation

---

## Vault Ontology (PARA Policy)

The Obsidian vault follows an explicit PARA-based ontology that is treated as **system policy**, not assumption:

- `03_projects/` → Active projects
- `02_areas/` → Ongoing responsibilities
- `04_resources/` → Reference material
- `05_archive/` → Inactive or completed items

Important rule:

Projects are considered active **only** if they are located under `03_projects/`.
File modification time is not used to determine activity.

This ontology is enforced via prompts and routing logic.

---

## Architecture Overview

```
User Query
   ↓
Router Agent (LLM Planner)
   ↓
Deterministic Tool Executor (Orchestrator)
   ↓
Answer Agent (LLM, grounded)
   ↓
Verifier (Deterministic checks)
```

---

## Agent Roles

### Router Agent
- Classifies user intent
- Outputs a structured JSON execution plan
- Selects tools but does not execute them

### Tool Executor (Orchestrator)
- Executes the plan exactly as produced
- Calls deterministic tools only
- Collects structured evidence

### Answer Agent
- Generates answers using only provided sources
- Enforces citation discipline
- Refuses to answer if evidence is insufficient

### Verifier
- Validates citation correctness
- Flags unsupported claims

---

## Supported Query Intents

| Intent        | Description                                  | Strategy                                  |
|---------------|----------------------------------------------|-------------------------------------------|
| RAG_QA        | Factual explanation or reasoning             | Semantic retrieval (FAISS, top-k chunks)  |
| DOC_EXTRACT   | Complete extraction from a specific document | Full document read + deterministic parse  |
| BROWSE        | Navigation or inventory of the vault         | Filesystem traversal                      |
| SYNTHESIS     | Multi-note reasoning and synthesis           | Planned extension                         |

---

## Tooling Layer

The system uses a deliberately small and orthogonal toolset:

- `search(query, top_k)`
  Semantic retrieval over FAISS index

- `resolve_note(name)`
  Fuzzy resolution of note names to vault paths

- `read_note(rel_path)`
  Reads full markdown content of a note

- `extract_resources(markdown)`
  Deterministic extraction of links and list items by heading

- `browse_vault(prefix, depth, ...)`
  Generic filesystem browsing

Folder semantics (e.g. active projects) are enforced via prompts, not hardcoded tools.

---

## Retrieval and Indexing

- Recursive vault ingestion
- Markdown split by headings and chunked by size
- Metadata preserved (path, heading, hash, timestamps)
- Embeddings generated using Sentence-Transformers
- FAISS index built with normalized vectors
- Manifest saved for reproducibility

Indexing is performed offline and reused at runtime.

---

## Fine-Tuning Strategy

Fine-tuning is applied using **LoRA / PEFT** and is explicitly **not** used for knowledge injection.

### Fine-Tuning Targets

- Router Agent
  - Intent classification accuracy
  - JSON plan stability
  - Tool selection consistency

- Answer Agent
  - Citation discipline
  - Refusal behavior
  - Structured, concise answers

Fine-tuning is used for **behavioral alignment**, not memorization.

---

## Evaluation

Evaluation is performed using a curated query set covering:

- RAG-based QA queries
- Document extraction completeness tests
- Vault browsing queries
- Adversarial queries requiring refusal

Metrics include:
- Citation correctness
- Completeness (for extraction tasks)
- Refusal accuracy
- Qualitative clarity

Results show reduced hallucination and improved task correctness compared to a naïve RAG baseline.

---

## Safety and Privacy

- Deterministic execution of all tools
- Mandatory source citations
- Explicit refusal when evidence is missing
- Post-generation verification
- Fully local execution
- No data leaves the user’s machine

---

## Setup Instructions

### 1. Set Environment Variable

```bash
export OBSIDIAN_VAULT_PATH=/path/to/your/ObsidianVault
```

### 2. Build the Index

```bash
python -m src.embed_index
```

### 3. Run the Assistant

```bash
python -m src.chat --q "your question here"
```

---

## Project Structure

```
src/
├── chat.py            # CLI entry point
├── orchestrator.py    # Tool execution engine
├── router_agent.py    # Intent planning agent
├── local_llm.py       # Local LLM wrapper
├── ingest.py          # Vault ingestion and chunking
├── embed_index.py     # Index building
├── retrieve.py        # FAISS retrieval
├── tools.py           # Deterministic tools
├── catalog.py         # Note catalog
├── prompts.py         # System and user prompts
├── models.py          # Data models
├── vault_policy.py    # PARA ontology
└── config.py          # Environment configuration
```

---

## Limitations and Future Work

- SYNTHESIS intent is limited and will be expanded
- Large documents may require extraction truncation
- UI integration with Obsidian is planned
- Richer verifier for semantic claim checking
- A Writer Agent

---

## License

This project is provided for academic and educational purposes.

---

## Author

Ahmed ELkhateeb

