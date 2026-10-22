
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")

if not all([uri, username, password]):
    raise ValueError("Neo4j configuration is missing from .env")

with GraphDatabase.driver(uri, auth=(username, password)) as driver:
    driver.verify_connectivity()
    with driver.session() as session:
        result = session.run(
            'RETURN "Neo4j connection successful" AS status'
        )
        print(result.single()["status"])
