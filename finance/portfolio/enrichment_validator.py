import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichmentValidator:
    REQUIRED_FIELDS = [
        "pe_ratio",
        "momentum_3m",
        "fii_flow_1m",
        "risk_flags",
        "sentiment_score"
    ]

    @staticmethod
    def is_ready(enriched_data: Dict[str, Any]) -> bool:
        """
        Validates that all required fields for every sector are populated.
        """
        if not enriched_data:
            logger.error("Enrichment validation failed: Data is empty.")
            return False

        for sector_name, data in enriched_data.items():
            for field in EnrichmentValidator.REQUIRED_FIELDS:
                if field not in data:
                    logger.error(f"Enrichment validation failed: Sector '{sector_name}' is missing required field '{field}'.")
                    return False
                
                # Special check for fields that shouldn't be N/A or None if possible
                # But as per user, some can be empty list (like risk_flags)
                val = data[field]
                if val is None:
                    logger.error(f"Enrichment validation failed: Sector '{sector_name}' has null value for '{field}'.")
                    return False
                
                # Check for "N/A" strings which might come from data_fetcher
                # pe_ratio is exempt — yfinance frequently can't fetch it for NSE indices
                if val == "N/A" and field != "pe_ratio":
                    logger.error(f"Enrichment validation failed: Sector '{sector_name}' has 'N/A' for '{field}'.")
                    return False

        logger.info("Enrichment validation passed: All required fields are present.")
        return True

if __name__ == "__main__":
    # Test validator
    test_data = {
        "Nifty IT": {
            "pe_ratio": 28.4,
            "momentum_3m": 5.1,
            "fii_flow_1m": -1200,
            "risk_flags": [],
            "sentiment_score": 0.5
        }
    }
    print(f"Validation Result: {EnrichmentValidator.is_ready(test_data)}")
