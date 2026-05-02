print("DEBUG: api/main.py loaded")
import sys
import os
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

print("[*] DeepChain API Starting...")
app = FastAPI(title="DeepChain Hybrid RAG API")
print("[+] FastAPI App Initialized")

# --- Monitoring disabled locally (prometheus causes startup hang) ---

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Initialization ---

# We use the same model across the stack
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# --- Lazy Initialization Helpers ---
# We wrap these to prevent the app from crashing on start (Status 1) if DBs are offline.

_hybrid_retriever = None

def get_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        try:
            from vector_store.weaviate_client import WeaviateClient
            from vector_store.embedder import GeminiEmbedder
            from vector_store.retriever import VectorRetriever
            from graph.neo4j_client import Neo4jClient
            w_client = WeaviateClient()
            emb = GeminiEmbedder()
            v_retriever = VectorRetriever(w_client, emb)
            n_client = Neo4jClient(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USERNAME", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password123")
            )
            from retrieval.hybrid_retriever import HybridRetriever
            _hybrid_retriever = HybridRetriever(
                retriever=v_retriever,
                neo4j_client=n_client,
                model_name=LLM_MODEL,
                top_k=5
            )
        except Exception as e:
            print(f"[!] Critical Error: Could not initialize RAG components: {e}")
            # We don't raise here, so the FastAPI app can still start and show health status
            return None
    return _hybrid_retriever

# --- Schemas ---

class QueryRequest(BaseModel):
    question: str
    method: str = "auto" # "naive", "graph", "hybrid", or "auto"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    method: str
    fallback_reason: str = ""
    latency: float = 0.0

# --- Finance Schemas ---

class PortfolioRequest(BaseModel):
    age: int
    monthly_income: float
    monthly_expenses: float
    pension: float = 0
    govt_allowances: float = 0
    additional_income: float = 0
    dependents: int = 0
    existing_savings: float = 0
    emergency_fund_exists: bool = False
    amount_to_invest: float = 0
    liabilities: list = []
    life_insurance: bool = False
    health_insurance: bool = False
    investment_horizon: str = "5yr"
    primary_goal: str = "Wealth Creation"

class TradeTestRequest(BaseModel):
    symbol: str
    strategy: str
    period: str = "1y"

class StrategyAdvisorRequest(BaseModel):
    intent: str

class MarketAdvisorRequest(BaseModel):
    symbol: str

# --- Routes ---

@app.get("/health")
async def health_check():
    """Check status of all dependent services (Weaviate, Neo4j)."""
    import socket

    def tcp_check(host, port, timeout=2):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    weaviate_host = os.getenv("WEAVIATE_HOST", "localhost")
    weaviate_port = int(os.getenv("WEAVIATE_PORT", 8080))
    neo4j_host    = os.getenv("NEO4J_HOST", "localhost")
    neo4j_port    = int(os.getenv("NEO4J_BOLT_PORT", 7687))

    weaviate_up = tcp_check(weaviate_host, weaviate_port)
    neo4j_up    = tcp_check(neo4j_host, neo4j_port)

    all_up = weaviate_up and neo4j_up
    return {
        "status":   "healthy" if all_up else "degraded",
        "services": {
            "weaviate": {
                "running": weaviate_up,
                "address": f"{weaviate_host}:{weaviate_port}",
                "fix":     None if weaviate_up else (
                    "docker run -d -p 8080:8080 -p 50051:50051 "
                    "cr.weaviate.io/semitechnologies/weaviate:latest"
                ),
            },
            "neo4j": {
                "running": neo4j_up,
                "address": f"{neo4j_host}:{neo4j_port}",
                "fix":     None if neo4j_up else (
                    "docker run -d -p 7474:7474 -p 7687:7687 "
                    "-e NEO4J_AUTH=neo4j/password neo4j:latest"
                ),
            },
        },
    }


@app.get("/metrics")
async def get_metrics():
    return {
        "status": "online",
        "uptime": "active",
        "engine": f"Gemini ({LLM_MODEL})",
        "pipeline": "Hybrid-RAG"
    }

@app.get("/health_v2")
def health_check_v2():
    retriever = get_retriever()
    if not retriever:
        return {"status": "degraded", "services": {"database": "offline"}}
    status = retriever.health_check()
    return {"status": "online", "services": status}

@app.post("/ingest")
def start_ingestion():
    """Triggers the document ingestion pipeline."""
    try:
        # Pre-flight check: is Weaviate reachable before we even try to connect?
        from vector_store.weaviate_client import WeaviateClient, WeaviateNotAvailableError
        if not WeaviateClient.is_available():
            raise WeaviateNotAvailableError(
                "Weaviate is not running. "
                "Start it with: docker run -d -p 8080:8080 -p 50051:50051 "
                "cr.weaviate.io/semitechnologies/weaviate:latest"
            )

        from ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        pipeline.run()
        return {"status": "success", "message": "Ingestion completed."}

    except WeaviateNotAvailableError as e:
        print(f"[!] Weaviate not available: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Weaviate vector database is not running.",
                "message": str(e),
                "fix": (
                    "Run this command to start Weaviate with Docker: "
                    "docker run -d -p 8080:8080 -p 50051:50051 "
                    "cr.weaviate.io/semitechnologies/weaviate:latest"
                ),
            },
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n[!!!] INGESTION ERROR:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    """Executes a RAG query using the specified method."""
    try:
        retriever = get_retriever()
        if not retriever:
            raise HTTPException(status_code=503, detail="Database services are currently unavailable.")
            
        # The new HybridRetriever handles retrieval + generation internally
        result = await retriever.query(
            question=request.question,
            mode=request.method,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result.answer,
            method=result.mode_used,
            fallback_reason=result.fallback_reason,
            latency=result.latency
        )
    except Exception as e:
        import traceback
        print(f"[!] API Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error during retrieval.")

# --- Finance Pipeline Routes ---

@app.post("/finance/portfolio")
async def get_portfolio_strategy(request: PortfolioRequest):
    """Generates a personalized portfolio allocation strategy."""
    try:
        from finance.portfolio.portfolio_pipeline import PortfolioPipeline
        pipeline = PortfolioPipeline()
        result = pipeline.run(request.dict())
        pipeline.close()
        
        # Flatten for frontend
        return {
            "status": result["strategy"]["status"],
            "risk_profile": result["strategy"]["risk_profile"],
            "allocations": result["strategy"]["allocations"],
            "explanation": result["explanation"],
            "surplus_income": result["strategy"]["health_status"]["monthly_surplus"],
            "is_fallback": result["strategy"].get("is_fallback", False)
        }
    except Exception as e:
        import traceback
        print(f"[!] Portfolio API Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Portfolio generation failed: {str(e)}")

@app.post("/finance/trade-test")
async def run_trade_test(request: TradeTestRequest):
    """Executes a backtest for a specific symbol and strategy."""
    try:
        from finance.trade_testing.trade_pipeline import TradeTestingPipeline
        pipeline = TradeTestingPipeline()
        result = pipeline.run_test(
            symbol=request.symbol,
            strategy_name=request.strategy,
            period=request.period
        )
        return result
    except Exception as e:
        import traceback
        print(f"[!] Trade Test API Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Trade test failed: {str(e)}")

@app.post("/finance/strategy-advisor")
async def get_strategy_advice(request: StrategyAdvisorRequest):
    """Generates a full trading strategy approach based on user intent."""
    try:
        from finance.strategies.advisor import StrategyAdvisor
        advisor = StrategyAdvisor()
        result  = await advisor.get_strategy_approach(request.intent)
        return {
            "approach_report":    result.get("approach_report", ""),
            "structured":         result.get("structured", {}),
            "retrieved_context":  result.get("retrieved_context", ""),
            "latency":            result.get("latency", 0.0),
            "model_used":         result.get("model_used", LLM_MODEL),
        }
    except Exception as e:
        import traceback
        print(f"[!] Strategy Advisor API Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Strategy generation failed: {str(e)}")

@app.post("/finance/market-advisor")
async def get_market_strategy_advice(request: MarketAdvisorRequest):
    """Generates a dynamic strategy based on live market conditions."""
    try:
        from finance.strategies.market_strategist import MarketStrategyAdvisor
        advisor = MarketStrategyAdvisor()
        result  = await advisor.analyze_and_build(request.symbol)
        return {
            "symbol":            result.get("symbol", request.symbol),
            "market_summary":    result.get("market_summary", {}),
            "dynamic_report":    result.get("dynamic_report", ""),
            "structured":        result.get("structured", {}),
            "retrieved_context": result.get("retrieved_context", ""),
            "model_used":        result.get("model_used", LLM_MODEL),
            "error":             result.get("error", ""),
        }
    except Exception as e:
        import traceback
        print(f"[!] Market Advisor API Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Market strategy generation failed: {str(e)}")

# ── ML Stock Strategy Advisory ─────────────────────────────────────────────────

class MLStockAdvisoryRequest(BaseModel):
    symbol: str

class MLTrainRequest(BaseModel):
    quick: bool = False


@app.post("/finance/ml-stock-advisory")
async def ml_stock_advisory(request: MLStockAdvisoryRequest):
    """
    Full ML-powered stock advisory:
    1. Fetches live OHLCV + fundamentals (PE, face value, mkt cap, 52w H/L, etc.)
    2. Computes 15 technical features
    3. ML model predicts best strategy + expected Sharpe
    4. Runs 2-year backtests for all 10 strategies
    5. If ML and backtest agree, use that strategy; if they disagree, auto-generate a Sharpe-weighted hybrid
    6. Returns full structured advisory with entry/exit levels
    """
    try:
        from finance.ml_engine.advisor_engine import StockStrategyAdvisor
        advisor = StockStrategyAdvisor(auto_train_quick=True)
        result  = advisor.advise(request.symbol)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        print(f"[!] ML Advisory Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ML advisory failed: {str(e)}")


@app.post("/finance/ml-train")
async def trigger_ml_training(request: MLTrainRequest):
    """
    Trigger ML model training on Nifty-50 symbols.
    quick=True: 10 symbols (~5 min). quick=False: 30 symbols (~25 min).
    """
    try:
        from finance.ml_engine.trainer import StrategyMLTrainer
        trainer = StrategyMLTrainer()
        result  = trainer.train(quick=request.quick)
        return {"status": "success", "training_summary": result}
    except Exception as e:
        import traceback
        print(f"[!] ML Training Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")


@app.get("/finance/ml-model-status")
async def ml_model_status():
    """Check if trained ML models exist and when they were last trained."""
    from pathlib import Path
    import json as _json
    model_dir  = Path("models")
    log_path   = model_dir / "training_log.json"
    clf_exists = (model_dir / "strategy_classifier.joblib").exists()
    reg_exists = (model_dir / "sharpe_regressor.joblib").exists()
    trained_at = None
    summary    = {}
    if log_path.exists():
        try:
            with open(log_path) as f:
                log = _json.load(f)
            trained_at = log.get("trained_at")
            summary    = log
        except Exception:
            pass
    return {
        "models_ready":      clf_exists and reg_exists,
        "classifier_exists": clf_exists,
        "regressor_exists":  reg_exists,
        "trained_at":        trained_at,
        "training_summary":  summary,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
