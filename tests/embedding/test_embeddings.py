import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from slopo.config import Config
from slopo.embedding.embeddings import EmbeddingError, embed_units, split_unsupported
from slopo.embedding.models import EmbeddedUnit, UnembeddedUnit

_CONFIG = Config(
    source_dir=Path("src"),
    source_dir_exclude=[],
    include_file_extensions=[],
    db_file=Path("slopo.db"),
    report_dir=Path("slopo-report"),
    ignore_file=Path("slopo.ignore.txt"),
    agent_log_file=Path("slopo.log"),
    embedding_model="openai/text-embedding-3-small",
    embedding_dimensions=3,
    embedding_api_key="test-key",
    embedding_params={},
    embedding_batch_size=100,
    embedding_batch_chars=10000,
    embedding_request_delay=0,
    analyze_similarity_threshold=0.9,
    analyze_rerank_threshold=0.93,
    review_similarity_threshold=0.9,
    review_rerank_threshold=0.93,
    body_node_count_threshold=10,
    block_node_count_threshold=30,
)


def _mock_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [{"embedding": v} for v in vectors]
    return response


def test_single_unit_mapped():
    units = [UnembeddedUnit(body_hash="h7", body="def foo(): pass")]
    with patch(
        "litellm.embedding",
        return_value=_mock_response([[1.0, 2.0, 3.0]]),
    ):
        result = embed_units(units, _CONFIG)

    assert result == [EmbeddedUnit(body_hash="h7", vector=[1.0, 2.0, 3.0])]


def test_multiple_units_preserve_order():
    units = [
        UnembeddedUnit(body_hash="h1", body="def foo(): pass"),
        UnembeddedUnit(body_hash="h2", body="def bar(): pass"),
        UnembeddedUnit(body_hash="h3", body="def baz(): pass"),
    ]
    with patch(
        "litellm.embedding",
        return_value=_mock_response(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    ):
        result = embed_units(units, _CONFIG)

    assert result == [
        EmbeddedUnit(body_hash="h1", vector=[1.0, 0.0, 0.0]),
        EmbeddedUnit(body_hash="h2", vector=[0.0, 1.0, 0.0]),
        EmbeddedUnit(body_hash="h3", vector=[0.0, 0.0, 1.0]),
    ]


def test_embedding_params_forwarded_to_litellm():
    units = [UnembeddedUnit(body_hash="h1", body="def foo(): pass")]
    config = replace(
        _CONFIG,
        embedding_params={"input_type": "search_document", "truncation": False},
    )
    with patch(
        "litellm.embedding",
        return_value=_mock_response([[1.0, 2.0, 3.0]]),
    ) as mock_embedding:
        embed_units(units, config)

    kwargs = mock_embedding.call_args.kwargs
    assert kwargs["input_type"] == "search_document"
    assert kwargs["truncation"] is False


def test_vector_size_mismatch_raises():
    units = [UnembeddedUnit(body_hash="h1", body="def foo(): pass")]
    with patch(
        "litellm.embedding",
        return_value=_mock_response([[1.0, 2.0]]),
    ), pytest.raises(EmbeddingError):
        embed_units(units, _CONFIG)


# --- local fastembed provider ---

_FASTEMBED_CONFIG = replace(
    _CONFIG, embedding_model="fastembed/some/model", embedding_api_key=None
)


def _fake_fastembed(
    monkeypatch: pytest.MonkeyPatch, vectors: list[list[float]]
) -> MagicMock:
    module = ModuleType("fastembed")
    text_embedding = MagicMock()
    text_embedding.return_value.embed.return_value = iter(vectors)
    module.TextEmbedding = text_embedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    monkeypatch.setattr("slopo.embedding.embeddings._fastembed_models", {})
    return text_embedding


def test_fastembed_embeds_locally_and_never_calls_litellm(
    monkeypatch: pytest.MonkeyPatch,
):
    text_embedding = _fake_fastembed(monkeypatch, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    units = [
        UnembeddedUnit(body_hash="h1", body="def foo(): pass"),
        UnembeddedUnit(body_hash="h2", body="def bar(): pass"),
    ]

    with patch("litellm.embedding") as litellm_embedding:
        result = embed_units(units, _FASTEMBED_CONFIG)

    litellm_embedding.assert_not_called()
    text_embedding.assert_called_once_with("some/model")
    assert result == [
        EmbeddedUnit(body_hash="h1", vector=[1.0, 2.0, 3.0]),
        EmbeddedUnit(body_hash="h2", vector=[4.0, 5.0, 6.0]),
    ]


def test_fastembed_loads_the_model_once_across_batches(monkeypatch: pytest.MonkeyPatch):
    text_embedding = _fake_fastembed(monkeypatch, [])
    text_embedding.return_value.embed.side_effect = lambda bodies: iter(
        [[0.0, 0.0, 0.0]] * len(bodies)
    )
    unit = UnembeddedUnit(body_hash="h1", body="def foo(): pass")

    embed_units([unit], _FASTEMBED_CONFIG)
    embed_units([unit], _FASTEMBED_CONFIG)

    assert text_embedding.call_count == 1


def test_fastembed_vector_size_mismatch_raises(monkeypatch: pytest.MonkeyPatch):
    _fake_fastembed(monkeypatch, [[1.0, 2.0]])
    units = [UnembeddedUnit(body_hash="h1", body="def foo(): pass")]

    with pytest.raises(EmbeddingError, match="size 2"):
        embed_units(units, _FASTEMBED_CONFIG)


def test_fastembed_not_installed_raises_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # import raises ImportError
    units = [UnembeddedUnit(body_hash="h1", body="def foo(): pass")]

    with pytest.raises(EmbeddingError, match="not installed"):
        embed_units(units, _FASTEMBED_CONFIG)


def test_split_unsupported_sets_aside_bodies_over_2000_characters_for_fastembed():
    fits = UnembeddedUnit(body_hash="fits", body="x" * 2000)
    too_long = UnembeddedUnit(body_hash="long", body="x" * 2001)

    supported, skipped = split_unsupported([fits, too_long], _FASTEMBED_CONFIG)

    assert (supported, skipped) == ([fits], [too_long])


def test_split_unsupported_sets_nothing_aside_for_api_providers():
    too_long = UnembeddedUnit(body_hash="long", body="x" * 50_000)

    assert split_unsupported([too_long], _CONFIG) == ([too_long], [])
