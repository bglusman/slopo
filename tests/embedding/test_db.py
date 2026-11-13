import sqlite3

import numpy as np
import pytest

from slopo.embedding.db import (
    count_unembedded_units,
    load_embeddings,
    load_next_batch,
    mark_skipped_embeddings,
    save_embeddings,
)
from slopo.embedding.models import EmbeddedUnit

# --- count_unembedded_units ---


def test_count_unembedded_units_ignores_embedded_hashes(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'b', 'b', 1, 2, 3, 'hashB');
        INSERT INTO embeddings (body_hash, embedding) VALUES ('hashB', X'0000803f');
    """)

    assert count_unembedded_units(conn) == 1


def test_count_unembedded_units_counts_each_hash_once(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'a', 'a', 1, 2, 3, 'hashA');
    """)

    assert count_unembedded_units(conn) == 1


# --- load_next_batch ---


def test_load_next_batch_skips_already_embedded_hashes(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'b', 'b', 1, 2, 3, 'hashB');
        INSERT INTO embeddings (body_hash, embedding) VALUES ('hashA', X'0000803f');
    """)

    batch = load_next_batch(conn, max_items=10, max_chars=10_000)

    assert [u.body_hash for u in batch] == ["hashB"]


def test_load_next_batch_emits_each_hash_once(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'a', 'a', 1, 2, 3, 'hashA');
    """)

    batch = load_next_batch(conn, max_items=10, max_chars=10_000)

    assert [u.body_hash for u in batch] == ["hashA"]


# --- load_embeddings ---


def test_load_embeddings_for_all_embedded_units(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (3, 1, 'b', 'b', 1, 2, 3, 'hashB');
        INSERT INTO embeddings (body_hash, embedding) VALUES ('hashA', X'0000803f');
    """)

    assert sorted(load_embeddings(conn)) == [1, 2]


# --- save_embeddings ---


def test_save_then_load_embeddings_round_trips_the_vector(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
    """)

    save_embeddings(conn, [EmbeddedUnit(body_hash="hashA", vector=[1.0, 2.0, 3.0])])

    loaded = load_embeddings(conn)
    assert np.array_equal(loaded[1], np.array([1.0, 2.0, 3.0], dtype=np.float32))


# --- skipped_embeddings ---


def _two_units(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'a', 'a', 1, 2, 3, 'hashA');
        INSERT INTO code_units (id, file_id, context, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'b', 'b', 1, 2, 3, 'hashB');
    """)


def test_skipped_hashes_are_neither_counted_nor_loaded(conn: sqlite3.Connection):
    _two_units(conn)
    mark_skipped_embeddings(conn, ["hashA"], "too long")

    assert count_unembedded_units(conn) == 1
    assert [u.body_hash for u in load_next_batch(conn, 10, 10_000)] == ["hashB"]


def test_marking_a_hash_skipped_twice_keeps_the_first_reason(conn: sqlite3.Connection):
    mark_skipped_embeddings(conn, ["hashA"], "first")
    mark_skipped_embeddings(conn, ["hashA"], "second")

    assert conn.execute(
        "SELECT body_hash, reason FROM skipped_embeddings"
    ).fetchall() == [("hashA", "first")]


def test_skipped_embeddings_refuses_a_null_hash(conn: sqlite3.Connection):
    # SQLite lets NULL into a TEXT PRIMARY KEY, and one NULL would make every
    # `body_hash NOT IN (SELECT body_hash FROM skipped_embeddings)` match nothing.
    with pytest.raises(sqlite3.IntegrityError):
        mark_skipped_embeddings(conn, [None], "x")  # type: ignore[list-item]


def test_save_embeddings_keeps_the_first_vector_for_a_repeated_hash(
    conn: sqlite3.Connection,
):
    save_embeddings(conn, [EmbeddedUnit(body_hash="hashA", vector=[1.0])])
    save_embeddings(conn, [EmbeddedUnit(body_hash="hashA", vector=[2.0])])

    rows = conn.execute("SELECT body_hash, embedding FROM embeddings").fetchall()
    assert [(h, np.frombuffer(b, dtype=np.float32).tolist()) for h, b in rows] == [
        ("hashA", [1.0])
    ]
