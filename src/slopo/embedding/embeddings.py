from slopo.config import Config
from slopo.embedding.models import UnembeddedUnit, EmbeddedUnit

_fastembed_cache: dict = {}


def embed_units(batch: list[UnembeddedUnit], config: Config) -> list[EmbeddedUnit]:
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
    return [
        EmbeddedUnit(body_hash=unit.body_hash, vector=item["embedding"])
        for unit, item in zip(batch, response.data)
    ]


def _embed_fastembed(
    batch: list[UnembeddedUnit], model_name: str, config: Config
) -> list[EmbeddedUnit]:
    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    if model_name not in _fastembed_cache:
        _fastembed_cache[model_name] = TextEmbedding(model_name)
    model = _fastembed_cache[model_name]

    vectors = list(model.embed([u.body for u in batch]))
    return [
        EmbeddedUnit(body_hash=unit.body_hash, vector=list(map(float, vec)))
        for unit, vec in zip(batch, vectors)
    ]
