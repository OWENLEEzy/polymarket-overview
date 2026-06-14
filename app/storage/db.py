import sqlite3
from pathlib import Path


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.execute("pragma foreign_keys = on")
    return db


def migrate(db: sqlite3.Connection) -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    db.executescript(schema_path.read_text())
    db.commit()
