import os

from slopo.config import Config
from slopo.embedding.models import EmbeddedUnit, UnembeddedUnit


class EmbeddingError(Exception):
    pass


_FASTEMBED_PREFIX = "fastembed/"
# The local ONNX models fail on long inputs, so longer bodies are not embedded.
_FASTEMBED_CHAR_LIMIT = 2000
_fastembed_models: dict = {}


def split_unsupported(
    batch: list[UnembeddedUnit], config: Config
) -> tuple[list[UnembeddedUnit], list[UnembeddedUnit]]:
    """Split a batch into the bodies the configured provider can embed and the rest.

    Only the local fastembed provider has a limit; API providers take everything.
    """
    if not config.embedding_model.startswith(_FASTEMBED_PREFIX):
        return batch, []
    supported = [u for u in batch if len(u.body) <= _FASTEMBED_CHAR_LIMIT]
    skipped = [u for u in batch if len(u.body) > _FASTEMBED_CHAR_LIMIT]
    return supported, skipped


def _embed_fastembed(batch: list[UnembeddedUnit], config: Config) -> list[EmbeddedUnit]:
    # Runs in-process: no API key and no server, and no code leaves the machine.
    try:
        from fastembed import TextEmbedding  # type: ignore[import-not-found]
    except ImportError as e:
        raise EmbeddingError(
            "embedding_model starts with 'fastembed/' but the fastembed package is not installed."
        ) from e

    model_name = config.embedding_model.removeprefix(_FASTEMBED_PREFIX)
    if model_name not in _fastembed_models:
        _fastembed_models[model_name] = TextEmbedding(model_name)
    vectors = _fastembed_models[model_name].embed([u.body for u in batch])

    result = []
    for unit, vector in zip(batch, vectors):
        values = [float(x) for x in vector]
        if len(values) != config.embedding_dimensions:
            raise EmbeddingError(
                f"Embedding model '{config.embedding_model}' returned a vector of "
                f"size {len(values)}, but embedding_dimensions is configured as "
                f"{config.embedding_dimensions}."
            )
        result.append(EmbeddedUnit(body_hash=unit.body_hash, vector=values))
    return result


def embed_units(batch: list[UnembeddedUnit], config: Config) -> list[EmbeddedUnit]:
    if config.embedding_model.startswith(_FASTEMBED_PREFIX):
        return _embed_fastembed(batch, config)

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import litellm  # type: ignore[import-untyped]

    litellm.suppress_debug_info = True
    try:
        response = litellm.embedding(  # type: ignore[call-overload]
            model=config.embedding_model,
            input=[u.body for u in batch],
            dimensions=config.embedding_dimensions,
            api_key=config.embedding_api_key,
            **config.embedding_params,
        )
    except tuple(litellm.LITELLM_EXCEPTION_TYPES) as e:  # type: ignore[misc]
        raise EmbeddingError(str(e)) from e

    result = []
    for unit, item in zip(batch, response.data):
        vector = item["embedding"]
        if len(vector) != config.embedding_dimensions:
            raise EmbeddingError(
                f"Embedding model '{config.embedding_model}' returned a vector of "
                f"size {len(vector)}, but embedding_dimensions is configured as "
                f"{config.embedding_dimensions}."
            )
        result.append(EmbeddedUnit(body_hash=unit.body_hash, vector=vector))
    return result
