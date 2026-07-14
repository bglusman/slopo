import sqlite3

from slopo.config import Config
from slopo.embedding.db import (
    count_unembedded_units,
    load_next_batch,
    mark_skipped_embeddings,
    save_embeddings,
)
from slopo.embedding.embeddings import embed_units
from slopo.progress import ProgressReporter


def run_embed(
    conn: sqlite3.Connection,
    cfg: Config,
    log: ProgressReporter,
) -> None:
    total = count_unembedded_units(conn)
    if total == 0:
        log("No new code units to embed.")
        return

    log("Starting...")

    embedded = 0
    while True:
        batch = load_next_batch(
            conn, cfg.embedding_batch_size, cfg.embedding_batch_chars
        )
        if not batch:
            break
        result = embed_units(batch, cfg)
        with conn:
            save_embeddings(conn, result.embeddings)
            mark_skipped_embeddings(
                conn,
                [unit.body_hash for unit in result.skipped],
                "unsupported by local FastEmbed provider",
            )
        embedded += len(result.embeddings)
        log(f"Embedded {embedded}/{total} code units...")
        if result.skipped:
            log(f"Skipped {len(result.skipped)} unsupported code units.")
    log("Done")
