"""
graph_db.py  -  the ONLY file that talks to the Neo4j driver.

Everything else calls  db.run(query, **params)  and gets back a list of plain dicts.
That keeps all other Layer 2 code independent of the driver.

    from graph_db import connect
    with connect() as db:
        rows = db.run("MATCH (s:Scheme) RETURN count(s) AS n")

connect() checks the connection and turns the driver's errors into plain-English messages.
"""
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ClientError, ConfigurationError, ServiceUnavailable

import kg_config as config
import net_check


class KGConnectionError(Exception):
    """Neo4j could not be reached or refused the login. The message says what to check."""


class Neo4jDB:
    def __init__(self, uri: str, user: str, password: str, database: str):
        self.uri, self.database = uri, database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def run(self, query: str, **params) -> list:
        """Run one Cypher query and return its rows as a list of dicts."""
        result = self._driver.execute_query(query, parameters_=params, database_=self.database)
        return [record.data() for record in result.records]

    def verify(self):
        """Raise KGConnectionError (with advice) if Neo4j cannot be used."""
        try:
            self._driver.verify_connectivity()
            self.run("RETURN 1 AS ok")
        except AuthError as e:
            raise KGConnectionError(
                "Neo4j rejected the username or password. Check NEO4J_USER and NEO4J_PASSWORD in "
                f"{config.ENV_FILE}. (For Aura, use the credentials file you downloaded.)") from e
        except ServiceUnavailable as e:
            failed = [r for r in net_check.diagnose(self.uri) if not r["ok"]]
            if failed:
                what = f"{failed[0]['stage']} stage failed: {failed[0]['detail']}. {failed[0]['advice']}"
            else:
                what = ("The network path looks fine, so the database itself is not ready. Neo4j Desktop: start "
                        "the database (it must say Active). Aura: make sure the instance says Running (wait a few "
                        "minutes after Resume).")
            raise KGConnectionError(f"Cannot reach Neo4j at {self.uri}. {what} (driver message: {e})") from e
        except ConfigurationError as e:
            raise KGConnectionError(f"NEO4J_URI looks invalid ({self.uri}). Details: {e}") from e
        except ClientError as e:
            if "DatabaseNotFound" in str(getattr(e, "code", "")):
                raise KGConnectionError(
                    f"Database '{self.database}' does not exist on this server. Set NEO4J_DATABASE in "
                    ".env to the name of your database (the default is 'neo4j').") from e
            raise KGConnectionError(f"Neo4j refused the request: {e}") from e

    def close(self):
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect() -> Neo4jDB:
    """Open and verify a connection using the settings in .env."""
    if config.ENV_PROBLEM:
        raise KGConnectionError(config.ENV_PROBLEM)
    if not config.NEO4J_PASSWORD:
        raise KGConnectionError(
            f"NEO4J_PASSWORD was not found. Add  NEO4J_PASSWORD=your-password  to {config.ENV_FILE} "
            "(saved as UTF-8).")
    db = Neo4jDB(config.NEO4J_URI, config.NEO4J_USER, config.NEO4J_PASSWORD, config.NEO4J_DATABASE)
    try:
        db.verify()
    except KGConnectionError:
        db.close()
        raise
    return db