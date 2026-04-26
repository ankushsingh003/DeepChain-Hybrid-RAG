"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Ragas Evaluation Suite - Benchmarking Model Performance
"""

import os
import json
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class RagasEvaluator:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def run_evaluation(self, test_data: List[Dict[str, Any]]) -> Dict:
        """Runs Ragas evaluation on a list of question/answer/context triplets."""
        print(f"[*] Starting Evaluation on {len(test_data)} samples...")
        
        # Prepare dataset for Ragas
        # test_data should be: [{"question": "...", "answer": "...", "contexts": ["..."], "ground_truth": "..."}]
        dataset = Dataset.from_list(test_data)
        
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
        )
        
        print("[+] Evaluation Complete.")
        return result.to_pandas().to_dict()

if __name__ == "__main__":
    # Sample Mock Evaluation
    evaluator = RagasEvaluator()
    sample_data = [{
        "question": "What is DeepChain?",
        "answer": "DeepChain is a hybrid RAG system.",
        "contexts": ["DeepChain combines knowledge graphs and vector search."],
        "ground_truth": "DeepChain is a hybrid RAG system using Neo4j and Weaviate."
    }]
    # stats = evaluator.run_evaluation(sample_data)
    # print(stats)
