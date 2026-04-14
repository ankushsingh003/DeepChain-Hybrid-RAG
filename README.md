# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence 🕸️🤖

DeepChain is an industry-level Knowledge Graph + RAG hybrid system designed for complex domains like Finance, Legal, and Healthcare. It leverages the structured context of **Neo4j** and the semantic power of **Weaviate** to provide highly faithful and transparent answers.

---

## 🚀 Key Features
- **Hybrid Retrieval**: Combines Neo4j (structural facts) + Weaviate (semantic chunks).
- **GraphRAG Implementation**: Community-driven triplet extraction using LLMs.
- **Enterprise Observability**: MLflow for experiment tracking & Prometheus/Grafana for monitoring.
- **Advanced Evaluation**: Ragas-based benchmarking (Naive RAG vs. GraphRAG).
- **Modern Tech Stack**: FastAPI, LangChain, Gemini 1.5 Pro, Docker, and Streamlit.

---

## 🛠️ Architecture
![Architecture Diagram](https://your-diagram-link-here.com)
1. **Ingestion**: PDFs/Txt → Semantic Chunking → Triplet Extraction (Subject-Predicate-Object).
2. **Storage**: Entities/Relations in Neo4j, Chunks in Weaviate.
3. **Retrieval**: 
   - **Naive**: Vector similarity search.
   - **GraphRAG**: Entity-anchored sub-graph retrieval + Vector context fusion.
4. **Serving**: FastAPI backend with Streamlit frontend.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Google Gemini API Key (or OpenAI/Groq)

### 2. Infrastructure Setup
```bash
docker-compose up -d
```
Starts: Neo4j (7474), Weaviate (8080), MLflow (5000), Prometheus, and Grafana.

### 3. Application Setup
```bash
# Clone the repo
git clone https://github.com/ankushsingh003/DeepChain-Hybrid-RAG.git
cd DeepChain-Hybrid-RAG

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# [EDIT .env with your keys]
```

---

## 🏗️ Usage

### Ingestion Pipeline
```bash
python -m ingestion.pipeline
```

### Running the API & UI
```bash
# Start Backend
python -m api.main

# Start Frontend
streamlit run ui/app.py
```

---

## 📊 Evaluation (Ragas)
Compare performance metrics between different RAG strategies:
```bash
python -m evaluation.benchmark
```

---

## 🤝 Contributing
Contributions are welcome for adding new graph traversal strategies or reranking algorithms!

## 📄 License
MIT License
