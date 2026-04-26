"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Neo4j Butler - Connection and Query Interface
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jClient:
    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password123")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def query(self, cypher: str, parameters: dict = None):
        """Executes a Cypher query and returns the results."""
        with self.driver.session() as session:
            result = session.run(cypher, parameters)
            return [record for record in result]

    def reset_db(self):
        """Clears the entire database (Development only)."""
        print("[!] Resetting Neo4j Database...")
        self.query("MATCH (n) DETACH DELETE n")

    def initialize_schema(self):
        """Creates indexes and constraints for efficient traversal."""
        print("[*] Initializing Neo4j Schema (Constraints/Indexes)...")
        # Ensure unique entity names
        try:
            self.query("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            print("[+] Neo4j Schema initialized.")
        except Exception as e:
            print(f"[!] Warning during schema init: {e}")

if __name__ == "__main__":
    # Test connection
    client = Neo4jClient()
    try:
        client.initialize_schema()
        print("[SUCCESS] Connected to Neo4j.")
    except Exception as e:
        print(f"[ERROR] Could not connect to Neo4j: {e}")
    finally:
        client.close()
