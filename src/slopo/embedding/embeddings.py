from slopo.config import Config
from slopo.embedding.models import EmbeddingResult, EmbeddedUnit, UnembeddedUnit

_fastembed_cache: dict = {}


def embed_units(batch: list[UnembeddedUnit], config: Config) -> EmbeddingResult:
    model_name = config.embedding_model

    # fastembed runs locally; LiteLLM 1.x doesn't route this prefix natively.
    if model_name.startswith("fastembed/"):
        return _embed_fastembed(batch, model_name.removeprefix("fastembed/"), config)

    import litellm  # type: ignore[import-untyped]

    response = litellm.embedding(
        model=model_name,
        input=[u.body for u in batch],
        dimensions=config.embedding_dimensions,
        api_key=config.embedding_api_key,
    )
    return EmbeddingResult(
        embeddings=[
            EmbeddedUnit(body_hash=unit.body_hash, vector=item["embedding"])
            for unit, item in zip(batch, response.data)
        ],
        skipped=[],
    )


_FASTEMBED_CHAR_LIMIT = 2000


def _embed_fastembed(
    batch: list[UnembeddedUnit], model_name: str, config: Config
) -> EmbeddingResult:
    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    safe = [unit for unit in batch if len(unit.body) <= _FASTEMBED_CHAR_LIMIT]
    skipped = [unit for unit in batch if len(unit.body) > _FASTEMBED_CHAR_LIMIT]
    if not safe:
        return EmbeddingResult(embeddings=[], skipped=skipped)

    if model_name not in _fastembed_cache:
        _fastembed_cache[model_name] = TextEmbedding(model_name)
    model = _fastembed_cache[model_name]

    vectors = list(model.embed([unit.body for unit in safe]))
    return EmbeddingResult(
        embeddings=[
            EmbeddedUnit(body_hash=unit.body_hash, vector=list(map(float, vec)))
            for unit, vec in zip(safe, vectors)
        ],
        skipped=skipped,
    )
