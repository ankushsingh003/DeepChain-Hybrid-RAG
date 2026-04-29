import os
import sys
import logging
import json
from typing import Dict, Any, List

# Add project root to path to import from existing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from graph.neo4j_client import Neo4jClient
from vector_store.weaviate_client import WeaviateClient
from vector_store.embedder import GeminiEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphEnricher:
    def __init__(self):
        try:
            self.neo4j = Neo4jClient()
            self.weaviate = WeaviateClient()
            self.embedder = GeminiEmbedder()
            logger.info("GraphEnricher initialized with Neo4j, Weaviate, and GeminiEmbedder.")
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            self.neo4j = None
            self.weaviate = None
            self.embedder = None

    def get_sector_entities(self, sector_name: str) -> List[str]:
        """
        Queries Neo4j for entities (companies) related to a sector.
        """
        if not self.neo4j: return []
        
        # Example cypher query to find companies in a sector
        # Assumes schema where (e:Entity {name: "Company Name"})-[:IN_SECTOR]->(s:Sector {name: "Sector Name"})
        # Or entities that are frequently mentioned with the sector.
        cypher = """
        MATCH (e:Entity)-[:PART_OF|ASSOCIATED_WITH|IN_SECTOR]->(s:Entity {name: $sector})
        RETURN e.name as name
        """
        try:
            records = self.neo4j.query(cypher, {"sector": sector_name})
            return [r["name"] for r in records]
        except Exception as e:
            logger.warning(f"Neo4j query failed for {sector_name}: {e}")
            return []

    def get_risk_flags_and_sentiment(self, sector_name: str, entities: List[str]) -> Dict[str, Any]:
        """
        Queries Weaviate for risk flags and sentiment related to the sector and its entities.
        """
        if not self.weaviate or not self.embedder:
            return {"risk_flags": [], "sentiment_score": 0.5, "insights": []}

        # Build a search query
        query_text = f"Regulatory risks, market sentiment, and news for {sector_name} sector in India. "
        if entities:
            query_text += f"Focus on: {', '.join(entities[:5])}."

        try:
            vector = self.embedder.embed_query(query_text)
            results = self.weaviate.search(vector, limit=10)
            
            risk_flags = []
            sentiment_sum = 0.0
            count = 0
            insights = []

            for obj in results:
                content = obj.properties.get("content", "").lower()
                # Simple keyword-based risk detection from RAG results
                risk_keywords = ["risk", "regulatory", "tightening", "restriction", "penalty", "slowdown", "decline", "crisis"]
                for kw in risk_keywords:
                    if kw in content:
                        # Extract a snippet as a flag
                        flag = content[:100] + "..."
                        if flag not in risk_flags:
                            risk_flags.append(flag)
                        break
                
                # Mock sentiment analysis (usually you'd use an LLM here)
                # For now, we'll use a very simple heuristic or just a neutral score
                sentiment_sum += 0.5 # Neutral default
                count += 1
                
                insights.append(obj.properties.get("content")[:200] + "...")

            avg_sentiment = sentiment_sum / count if count > 0 else 0.5
            
            return {
                "risk_flags": risk_flags[:3], # Top 3 flags
                "sentiment_score": round(avg_sentiment, 2),
                "insights": insights[:3]
            }
        except Exception as e:
            logger.warning(f"Weaviate search failed for {sector_name}: {e}")
            return {"risk_flags": [], "sentiment_score": 0.5, "insights": []}

    def enrich_sector_data(self, sector_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enriches the raw sector data with graph and vector insights.
        """
        enriched_results = {}
        
        for sector_name, data in sector_data.items():
            logger.info(f"Enriching {sector_name}...")
            
            # 1. Get related entities from Neo4j
            entities = self.get_sector_entities(sector_name)
            
            # 2. Get risks and sentiment from Weaviate
            rag_context = self.get_risk_flags_and_sentiment(sector_name, entities)
            
            # 3. Merge
            enriched_sector = data.copy()
            enriched_sector.update({
                "risk_flags": rag_context["risk_flags"],
                "sentiment_score": rag_context["sentiment_score"],
                "key_entities": entities,
                "rag_insights": rag_context["insights"]
            })
            
            enriched_results[sector_name] = enriched_sector
            
        return enriched_results

    def close(self):
        if self.neo4j: self.neo4j.close()
        if self.weaviate: self.weaviate.close()

if __name__ == "__main__":
    # Mock data for testing
    mock_data = {
        "Nifty IT": {"pe_ratio": 28.4, "momentum_3m": 5.1, "fii_flow_1m": -1200},
        "Nifty Bank": {"pe_ratio": 18.2, "momentum_3m": 2.1, "fii_flow_1m": 800}
    }
    enricher = GraphEnricher()
    enriched = enricher.enrich_sector_data(mock_data)
    print(json.dumps(enriched, indent=4))
    enricher.close()
