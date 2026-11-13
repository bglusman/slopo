import sqlite3
import time

from slopo.config import Config
from slopo.embedding.db import (
    count_unembedded_units,
    load_next_batch,
    mark_skipped_embeddings,
    save_embeddings,
)
from slopo.embedding.embeddings import embed_units, split_unsupported
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

    processed = 0
    skipped_total = 0
    while True:
        batch = load_next_batch(
            conn, cfg.embedding_batch_size, cfg.embedding_batch_chars
        )
        if not batch:
            break
        supported, skipped = split_unsupported(batch, cfg)
        batch_embeddings = embed_units(supported, cfg) if supported else []
        with conn:
            save_embeddings(conn, batch_embeddings)
            mark_skipped_embeddings(
                conn,
                [unit.body_hash for unit in skipped],
                "longer than the local fastembed provider accepts",
            )
        processed += len(batch)
        skipped_total += len(skipped)
        note = f" ({skipped_total} skipped as too long)" if skipped_total else ""
        log(f"Embedded {processed}/{total} code units{note}...")
        if cfg.embedding_request_delay > 0:
            time.sleep(cfg.embedding_request_delay)
    log("Done")
