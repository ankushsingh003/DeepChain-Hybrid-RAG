"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: LLM-based Entity and Relationship Extractor (Triple Extraction)
"""

import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- Schema Definitions ---

class Entity(BaseModel):
    name: str = Field(description="Name of the entity (e.g., Novatech Solutions, John Doe)")
    type: str = Field(description="Category of the entity (e.g., Organization, Person, Date, Location, Concept)")
    description: str = Field(description="Brief context or description of the entity found in the text")

class Relationship(BaseModel):
    source: str = Field(description="The source entity name")
    target: str = Field(description="The target entity name")
    type: str = Field(description="The relationship type (e.g., OWNS, INVESTED_IN, LOCATED_AT, COMPETES_WITH)")
    description: str = Field(description="Context of the relationship")

class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(description="List of all extracted entities")
    relationships: List[Relationship] = Field(description="List of all extracted relationships")

# --- Extractor Implementation ---

class GraphExtractor:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        
        self.prompt = ChatPromptTemplate.from_template(
            "Extract entities and their relationships from the following text to build a knowledge graph.\n"
            "Focus specifically on facts relevant to business, finance, and legal domains.\n"
            "{format_instructions}\n"
            "Text: {text}\n"
        )

    def extract(self, text: str) -> KnowledgeGraph:
        """Extracts structured entities and relationships from raw text."""
        print("[*] Extracting entities and relationships using LLM...")
        _input = self.prompt.format_prompt(
            text=text, 
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            response = self.llm.invoke(_input.to_messages())
            return self.parser.parse(response.content)
        except Exception as e:
            print(f"[!] Extraction failed: {e}")
            return KnowledgeGraph(entities=[], relationships=[])

if __name__ == "__main__":
    # Test sample
    test_text = (
        "Novatech Solutions is a fintech leader based in Mumbai. "
        "It was founded by Rajesh Sharma in 2015. "
        "The company recently acquired FinPay for $200 million."
    )
    
    extractor = GraphExtractor()
    kg = extractor.extract(test_text)
    
    print("\n[Extracted Entities]:")
    for e in kg.entities:
        print(f" - {e.name} ({e.type}): {e.description}")
        
    print("\n[Extracted Relationships]:")
    for r in kg.relationships:
        print(f" - {r.source} --[{r.type}]--> {r.target}")
        
