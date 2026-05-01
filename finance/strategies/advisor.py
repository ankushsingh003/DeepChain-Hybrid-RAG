import os
import time
import logging
from typing import Dict, Any, List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class StrategyAdvisor:
    """
    The Strategy Advisor 'Model' that provides full trading strategy approaches.
    It uses RAG to retrieve established strategies and then generates custom code.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.retriever = None # Will be initialized on demand

    def _get_retriever(self):
        """Lazy init of the hybrid retriever."""
        if self.retriever is None:
            try:
                from retrieval.hybrid_retriever import HybridRetriever
                from vector_store.weaviate_client import WeaviateClient
                from vector_store.embedder import GeminiEmbedder
                from vector_store.retriever import VectorRetriever
                from graph.neo4j_client import Neo4jClient
                
                w_client = WeaviateClient()
                emb = GeminiEmbedder()
                v_retriever = VectorRetriever(w_client, emb)
                n_client = Neo4jClient()
                
                self.retriever = HybridRetriever(
                    retriever=v_retriever,
                    neo4j_client=n_client,
                    model_name="gemini-2.0-flash"
                )
            except Exception as e:
                logger.error(f"Could not initialize retriever for advisor: {e}")
                return None
        return self.retriever

    async def get_strategy_approach(self, user_intent: str) -> Dict[str, Any]:
        """
        Retrieves relevant strategy context and generates a full building approach.
        """
        start_time = time.time()
        
        # 1. Retrieve Knowledge
        retriever = self._get_retriever()
        context = ""
        if retriever:
            # We search specifically for the user intent in our knowledge base
            rag_result = await retriever.query(
                question=f"Which of the 10 trading strategies matches this intent: {user_intent}? Provide details.",
                mode="hybrid"
            )
            context = rag_result.answer

        # 2. Generate Full Strategy Approach
        prompt = f"""
        System: You are an expert Quant Trading Strategist for DeepChain.
        Context from Knowledge Base: {context}
        
        User Request: {user_intent}
        
        Task: Provide a 'Full Strategy and Trading Code Approach Building' report.
        The report MUST include:
        1. Strategy Overview: Detailed theory and logic.
        2. Mathematical Formulation: Equations for indicators.
        3. Implementation Code: Complete Python code (using pandas/numpy) for the strategy function.
        4. Entry/Exit Logic: Clear rules for signals.
        5. Backtesting Approach: How to validate this strategy.
        6. Risk Management: Proposed stop-loss and position sizing logic.
        
        Ensure the code follows the signature: `def strategy_function(df):` and returns a 'signals' column.
        Use a professional, premium tone.
        """
        
        try:
            response = self.model.generate_content(prompt)
            report = response.text
        except Exception as e:
            logger.error(f"Generation error: {e}")
            report = f"Error generating strategy: {str(e)}"

        return {
            "approach_report": report,
            "latency": time.time() - start_time,
            "retrieved_context": context
        }

if __name__ == "__main__":
    # Test
    import asyncio
    async def test():
        advisor = StrategyAdvisor()
        result = await advisor.get_strategy_approach("I want a strategy based on volatility and breakouts.")
        print(result["approach_report"])
    
    asyncio.run(test())
