# core/mongo_pool.py
"""
Shared MongoDB connection pool.
One client, created once, reused by all operations.
Replaces the pattern of MongoClient() + client.close() on every call.
"""

from pymongo import MongoClient
from urllib.parse import quote_plus
import logging

logger = logging.getLogger(__name__)

_client = None


def get_mongo_client() -> MongoClient:
    """Return the shared MongoClient (creates it on first call)."""
    global _client
    if _client is None:
        encoded_pass = quote_plus("LT@connect25")
        connection_string = (
            f"mongodb://connectly:{encoded_pass}"
            f"@192.168.48.201:27017/ml_notes"
            f"?authSource=admin"
            f"&maxPoolSize=50"
            f"&minPoolSize=5"
            f"&serverSelectionTimeoutMS=10000"
        )
        _client = MongoClient(connection_string)
        logger.info("✅ Shared MongoDB connection pool created (maxPoolSize=50)")
    return _client


def get_db():
    """Return the ml_notes database."""
    return get_mongo_client()["ml_notes"]


def close_pool():
    """Call on server shutdown to cleanly close all connections."""
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("✅ MongoDB connection pool closed")