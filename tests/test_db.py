import pytest

from slopo import db
from slopo.config import Config
from slopo.db import adopt_source_dir, chunked, create_db, open_db


def test_yields_nothing_for_empty_input():
    assert list(chunked([])) == []


def test_yields_single_chunk_when_input_fits(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "_MAX_SQL_VARIABLES", 3)

    assert list(chunked([1, 2])) == [[1, 2]]


def test_splits_input_across_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(db, "_MAX_SQL_VARIABLES", 2)

    assert list(chunked([1, 2, 3, 4, 5])) == [[1, 2], [3, 4], [5]]


def test_exact_multiple_leaves_no_trailing_empty_chunk(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(db, "_MAX_SQL_VARIABLES", 2)

    assert list(chunked([1, 2, 3, 4])) == [[1, 2], [3, 4]]


def test_adopt_source_dir_updates_metadata(tmp_path):
    original_source_dir = tmp_path / "old"
    adopted_source_dir = tmp_path / "new"
    original_source_dir.mkdir()
    adopted_source_dir.mkdir()
    db_file = tmp_path / "slopo.db"
    cfg = Config(
        source_dir=original_source_dir,
        source_dir_exclude=[],
        embedding_model="voyage/voyage-code-3",
        embedding_dimensions=1024,
        embedding_api_key="test-key",
        embedding_batch_size=100,
        embedding_batch_chars=100_000,
        similarity_threshold=0.92,
        rerank_threshold=0.94,
        body_node_count_threshold=10,
        db_file=db_file,
        report_dir=tmp_path / "report",
        ignore_file=tmp_path / "ignore.txt",
    )
    conn = create_db(cfg)

    previous = adopt_source_dir(conn, adopted_source_dir)
    stored = conn.execute("SELECT source_dir FROM metadata WHERE id = 1").fetchone()

    assert previous == str(original_source_dir.resolve())
    assert stored[0] == str(adopted_source_dir.resolve())


def test_open_db_migrates_v2_database_to_track_skipped_embeddings(tmp_path):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    db_file = tmp_path / "slopo.db"
    cfg = Config(
        source_dir=source_dir,
        source_dir_exclude=[],
        embedding_model="voyage/voyage-code-3",
        embedding_dimensions=1024,
        embedding_api_key="test-key",
        embedding_batch_size=100,
        embedding_batch_chars=100_000,
        similarity_threshold=0.92,
        rerank_threshold=0.94,
        body_node_count_threshold=10,
        db_file=db_file,
        report_dir=tmp_path / "report",
        ignore_file=tmp_path / "ignore.txt",
    )
    conn = create_db(cfg)
    conn.execute("DROP TABLE skipped_embeddings")
    conn.execute("UPDATE schema_version SET version = 2")
    conn.commit()
    conn.close()

    migrated = open_db(cfg)

    assert migrated.execute("SELECT version FROM schema_version").fetchone() == (3,)
    assert migrated.execute("SELECT name FROM sqlite_master WHERE name = 'skipped_embeddings'").fetchone()
