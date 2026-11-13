import sqlite3
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from slopo.config import Config
from slopo.embedding.command import run_embed
from slopo.embedding.models import EmbeddedUnit, UnembeddedUnit

_FASTEMBED_CONFIG = Config(
    source_dir=Path("src"),
    source_dir_exclude=[],
    include_file_extensions=[],
    db_file=Path("slopo.db"),
    report_dir=Path("slopo-report"),
    ignore_file=Path("slopo.ignore.txt"),
    agent_log_file=Path("slopo.log"),
    embedding_model="fastembed/some/model",
    embedding_dimensions=3,
    embedding_api_key=None,
    embedding_params={},
    embedding_batch_size=100,
    embedding_batch_chars=100_000,
    embedding_request_delay=0,
    analyze_similarity_threshold=0.9,
    analyze_rerank_threshold=0.93,
    review_similarity_threshold=0.9,
    review_rerank_threshold=0.93,
    body_node_count_threshold=10,
    block_node_count_threshold=30,
)


def _add_unit(
    conn: sqlite3.Connection, unit_id: int, body: str, body_hash: str
) -> None:
    conn.execute("INSERT OR IGNORE INTO files (id, path, mtime) VALUES (1, 'a.ex', 0)")
    conn.execute(
        "INSERT INTO code_units (id, file_id, context, body, start_line, end_line,"
        " body_node_count, body_hash) VALUES (?, 1, 'c', ?, 1, 2, 3, ?)",
        (unit_id, body, body_hash),
    )


def _fake_embed(batch: list[UnembeddedUnit], _config: Config) -> list[EmbeddedUnit]:
    return [EmbeddedUnit(body_hash=u.body_hash, vector=[1.0, 2.0, 3.0]) for u in batch]


def test_embed_saves_short_bodies_and_skips_long_ones_for_fastembed(
    conn: sqlite3.Connection,
):
    _add_unit(conn, 1, "def short, do: 1", "short")
    _add_unit(conn, 2, "x" * 2001, "long")
    logs: list[str] = []

    with patch("slopo.embedding.command.embed_units", side_effect=_fake_embed) as embed:
        run_embed(conn, _FASTEMBED_CONFIG, logs.append)

    assert [u.body_hash for u in embed.call_args.args[0]] == ["short"]
    assert [r[0] for r in conn.execute("SELECT body_hash FROM embeddings")] == ["short"]
    assert [r[0] for r in conn.execute("SELECT body_hash FROM skipped_embeddings")] == [
        "long"
    ]
    assert logs[-2:] == ["Embedded 2/2 code units (1 skipped as too long)...", "Done"]


def test_embed_terminates_when_a_batch_holds_only_unsupported_bodies(
    conn: sqlite3.Connection,
):
    _add_unit(conn, 1, "x" * 2001, "long")

    with patch("slopo.embedding.command.embed_units") as embed:
        run_embed(conn, _FASTEMBED_CONFIG, lambda _message: None)

    embed.assert_not_called()
    assert conn.execute("SELECT COUNT(*) FROM skipped_embeddings").fetchone() == (1,)


def test_embed_sends_every_body_to_api_providers(conn: sqlite3.Connection):
    _add_unit(conn, 1, "x" * 5000, "long")
    config = replace(_FASTEMBED_CONFIG, embedding_model="openai/text-embedding-3-small")

    with patch("slopo.embedding.command.embed_units", side_effect=_fake_embed) as embed:
        run_embed(conn, config, lambda _message: None)

    assert [u.body_hash for u in embed.call_args.args[0]] == ["long"]
    assert conn.execute("SELECT COUNT(*) FROM skipped_embeddings").fetchone() == (0,)
