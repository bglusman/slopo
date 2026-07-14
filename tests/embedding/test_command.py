import sqlite3
from pathlib import Path
from unittest.mock import patch

from slopo.config import Config
from slopo.embedding.command import run_embed
from slopo.embedding.models import EmbeddingResult, UnembeddedUnit


def test_run_embed_marks_unsupported_units_and_finishes(conn: sqlite3.Connection):
    conn.executescript("""
        INSERT INTO files (id, path, mtime) VALUES (1, 'File.java', 0);
        INSERT INTO code_units (id, file_id, name, body, start_line, end_line, body_node_count, body_hash)
            VALUES (1, 1, 'short', 'short body', 1, 2, 3, 'short');
        INSERT INTO code_units (id, file_id, name, body, start_line, end_line, body_node_count, body_hash)
            VALUES (2, 1, 'long', 'long body', 3, 4, 3, 'long');
    """)
    config = Config(
        source_dir=Path("src"), source_dir_exclude=[], db_file=Path("slopo.db"),
        report_dir=Path("slopo-report"), ignore_file=Path("slopo.ignore.txt"),
        embedding_model="fastembed/model", embedding_dimensions=3,
        embedding_api_key=None, embedding_batch_size=100, embedding_batch_chars=10_000,
        similarity_threshold=0.9, rerank_threshold=0.93, body_node_count_threshold=10,
    )
    result = EmbeddingResult(
        embeddings=[],
        skipped=[UnembeddedUnit(body_hash="short", body="short body"), UnembeddedUnit(body_hash="long", body="long body")],
    )

    with patch("slopo.embedding.command.embed_units", return_value=result) as embed:
        run_embed(conn, config, lambda _message: None)

    assert embed.call_count == 1
    assert conn.execute("SELECT body_hash FROM skipped_embeddings ORDER BY body_hash").fetchall() == [
        ("long",), ("short",)
    ]
