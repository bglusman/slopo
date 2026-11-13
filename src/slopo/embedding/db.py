import sqlite3

import numpy as np

from slopo.embedding.batching import trim_to_char_limit
from slopo.embedding.models import EmbeddedUnit, UnembeddedUnit


def count_unembedded_units(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT body_hash) FROM code_units
        WHERE body_hash NOT IN (SELECT body_hash FROM embeddings)
          AND body_hash NOT IN (SELECT body_hash FROM skipped_embeddings)
        """
    ).fetchone()
    return row[0]


def load_next_batch(
    conn: sqlite3.Connection, max_items: int, max_chars: int
) -> list[UnembeddedUnit]:
    rows = conn.execute(
        """
        SELECT body_hash, MIN(body)
        FROM code_units
        WHERE body_hash NOT IN (SELECT body_hash FROM embeddings)
          AND body_hash NOT IN (SELECT body_hash FROM skipped_embeddings)
        GROUP BY body_hash
        LIMIT ?
    """,
        (max_items,),
    ).fetchall()

    units = [UnembeddedUnit(body_hash=row[0], body=row[1]) for row in rows]
    return trim_to_char_limit(units, max_chars)


def save_embeddings(conn: sqlite3.Connection, embeddings: list[EmbeddedUnit]) -> None:
    conn.executemany(
        "INSERT INTO embeddings (body_hash, embedding) VALUES (?, ?)"
        " ON CONFLICT (body_hash) DO NOTHING",
        [
            (e.body_hash, np.asarray(e.vector, dtype=np.float32).tobytes())
            for e in embeddings
        ],
    )


def mark_skipped_embeddings(
    conn: sqlite3.Connection, body_hashes: list[str], reason: str
) -> None:
    conn.executemany(
        "INSERT INTO skipped_embeddings (body_hash, reason) VALUES (?, ?)"
        " ON CONFLICT (body_hash) DO NOTHING",
        [(body_hash, reason) for body_hash in body_hashes],
    )


def load_embeddings(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    rows = conn.execute(
        """
        SELECT cu.id, e.embedding
        FROM code_units cu
        JOIN embeddings e ON e.body_hash = cu.body_hash
        """
    ).fetchall()
    return {row[0]: np.frombuffer(row[1], dtype=np.float32) for row in rows}
