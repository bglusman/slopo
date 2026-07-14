from typing import NamedTuple


class UnembeddedUnit(NamedTuple):
    body_hash: str
    body: str


class EmbeddedUnit(NamedTuple):
    body_hash: str
    vector: list[float]


class EmbeddingResult(NamedTuple):
    embeddings: list[EmbeddedUnit]
    skipped: list[UnembeddedUnit]
