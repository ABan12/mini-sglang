from types import SimpleNamespace

import pytest
import torch
from minisgl.engine.engine import _adjust_config


def config(**overrides):
    values = dict(
        kv_cache_dtype="int8", attention_backend="auto", dtype=torch.bfloat16,
        model_config=SimpleNamespace(model_type="qwen3", is_moe=False),
        tp_info=SimpleNamespace(size=1), cache_type="radix", page_size=1,
        cuda_graph_bs=[1, 2, 4, 8, 16], cuda_graph_max_bs=16,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_int8_auto_keeps_cuda_graph_and_selects_triton():
    settings = config()
    _adjust_config(settings)
    assert settings.attention_backend == "quant-triton"
    assert settings.cache_type == "naive"
    assert settings.cuda_graph_bs == [1, 2, 4, 8, 16]
    assert settings.cuda_graph_max_bs == 16


def test_reference_remains_available_for_diagnostics():
    settings = config(attention_backend="quant-reference")
    _adjust_config(settings)
    assert settings.attention_backend == "quant-reference"
    assert settings.cuda_graph_bs == []
    assert settings.cuda_graph_max_bs == 0


@pytest.mark.parametrize("overrides,exception", [
    ({"kv_cache_dtype": "auto", "attention_backend": "quant-triton"}, ValueError),
    ({"dtype": torch.float16}, NotImplementedError),
    ({"tp_info": SimpleNamespace(size=2)}, NotImplementedError),
    ({"attention_backend": "fi"}, ValueError),
    ({"attention_backend": "fi,quant-triton"}, NotImplementedError),
])
def test_unsupported_combinations_fail_before_allocation(overrides, exception):
    with pytest.raises(exception):
        _adjust_config(config(**overrides))
