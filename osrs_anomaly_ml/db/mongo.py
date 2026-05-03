import os
from pymongo import MongoClient
from pymongo.collection import Collection

schema = None


def get_connection_string() -> str:
    return os.getenv("MONGODB_URI", "mongodb://localhost:27017")


def get_db():
    global schema

    if schema is None:
        connection_string = get_connection_string()
        client = MongoClient(connection_string)

        db_name = os.getenv("MONGODB_DB", "ml_models_db")
        db = client[db_name]

        categories = db["categories"]
        features = db["features"]
        rows = db["csv_rows"]

        rows.create_index([("file_id", 1)])
        rows.create_index([("feature_id", 1)])
        rows.create_index([("anomaly_count", -1)])
        rows.create_index([("anomaly_label", -1)])
        rows.create_index([("anomaly_score", -1)])

        schema = {
            "categories": categories,
            "features": features,
            "rows": rows,
            "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 500))
        }

    return schema


def categories() -> Collection:
    return get_db()["categories"]


def features() -> Collection:
    return get_db()["features"]


def rows() -> Collection:
    return get_db()["rows"]


CHUNK_SIZE = get_db()["CHUNK_SIZE"]