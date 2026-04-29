import logging
import json
from typing import Dict, Any

from .data_fetcher import fetch_sector_data
from .graph_enricher import GraphEnricher
from .enrichment_validator import EnrichmentValidator
from .strategy import PortfolioStrategy
from .explainer import PortfolioExplainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioPipeline:
    def __init__(self):
        self.enricher = GraphEnricher()
        self.validator = EnrichmentValidator()
        self.strategy = PortfolioStrategy()
        self.explainer = PortfolioExplainer()

    def run(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the full portfolio generation pipeline.
        """
        logger.info("--- Starting Portfolio Generation Pipeline ---")
        
        # Stage 1: Fetch Live Market Data
        logger.info("Stage 1: Fetching live market data...")
        raw_market_data = fetch_sector_data()
        
        # Stage 2: Graph Enrichment
        logger.info("Stage 2: Enriching data with Knowledge Graph & RAG...")
        enriched_data = self.enricher.enrich_sector_data(raw_market_data)
        
        # The Gate: Validation
        logger.info("Stage 2.5: Validating data completeness...")
        if not self.validator.is_ready(enriched_data):
            logger.error("Pipeline blocked by Validator Gate due to incomplete data.")
            return {
                "success": False,
                "error": "Data enrichment failed or incomplete. Strategy aborted.",
                "details": "Validator check failed."
            }
        
        # Stage 3: Strategy Execution
        logger.info("Stage 3: Running portfolio strategy engine...")
        strategy_results = self.strategy.calculate_allocation(user_profile, enriched_data)
        
        # Stage 4: Explanation
        logger.info("Stage 4: Generating AI explanation...")
        explanation = self.explainer.explain(user_profile, strategy_results)
        
        logger.info("--- Pipeline Execution Complete ---")
        
        return {
            "success": True,
            "profile": user_profile,
            "market_state": raw_market_data,
            "strategy": strategy_results,
            "explanation": explanation
        }

    def close(self):
        self.enricher.close()

if __name__ == "__main__":
    # Full System Integration Test
    pipeline = PortfolioPipeline()
    
    test_user = {
        "age": 28,
        "monthly_income": 120000,
        "monthly_expenses": 50000,
        "pension": 0,
        "govt_allowances": 0,
        "additional_income": 5000,
        "dependents": 0,
        "existing_savings": 200000,
        "emergency_fund_exists": True,
        "amount_to_invest": 300000,
        "liabilities": [
            {"name": "Personal Loan", "amount": 20000, "interest_rate": 14.5}
        ],
        "life_insurance": True,
        "health_insurance": True,
        "investment_horizon": "5yr",
        "primary_goal": "Wealth Creation"
    }
    
    result = pipeline.run(test_user)
    print("\n--- FINAL OUTPUT ---")
    print(json.dumps(result, indent=4))
    pipeline.close()
