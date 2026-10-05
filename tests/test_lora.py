import copy

import pytest
import torch
import torch.nn as nn

from src.lora import (
    ALL_TARGETS,
    ATTN_TARGETS,
    LoRALinear,
    count_parameters,
    inject_lora,
    load_lora,
    lora_modules,
    merge_and_unload,
    merge_lora,
    save_lora,
    unmerge_lora,
)


class TinyBlock(nn.Module):
    def __init__(self, d=16, hidden=32):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)
        self.gate_proj = nn.Linear(d, hidden)
        self.up_proj = nn.Linear(d, hidden)
        self.down_proj = nn.Linear(hidden, d)

    def forward(self, x):
        a = self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))
        h = torch.relu(self.gate_proj(x + a)) * self.up_proj(x + a)
        return x + a + self.down_proj(h)


class TinyModel(nn.Module):
    def __init__(self, n_layers=2, d=16):
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(d) for _ in range(n_layers)])
        self.lm_head = nn.Linear(d, 8)  # must NOT be wrapped

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


@pytest.fixture
def base_and_x():
    torch.manual_seed(0)
    model = TinyModel().eval()
    x = torch.randn(4, 5, 16)
    return model, x


def randomize_B(model):
    """After init B == 0; give it real values to test non-trivial updates."""
    for _, m in lora_modules(model):
        nn.init.normal_(m.lora_B, std=0.1)


def test_zero_init_keeps_model_identical(base_and_x):
    base, x = base_and_x
    ref = base(x)
    model = copy.deepcopy(base)
    inject_lora(model, ATTN_TARGETS, r=4, alpha=8)
    assert torch.allclose(model(x), ref, atol=1e-6)


def test_targets_and_lm_head_not_wrapped(base_and_x):
    base, _ = base_and_x
    model = copy.deepcopy(base)
    n = inject_lora(model, ATTN_TARGETS, r=4, alpha=8)
    assert n == 2 * 4  # 2 layers x 4 attention projections
    assert isinstance(model.layers[0].q_proj, LoRALinear)
    assert not isinstance(model.layers[0].gate_proj, LoRALinear)
    assert not isinstance(model.lm_head, LoRALinear)

    model2 = copy.deepcopy(base)
    assert inject_lora(model2, ALL_TARGETS, r=4, alpha=8) == 2 * 7


def test_trainable_param_count_matches_formula(base_and_x):
    base, _ = base_and_x
    model = copy.deepcopy(base)
    r = 4
    inject_lora(model, ATTN_TARGETS, r=r, alpha=8)
    trainable, total = count_parameters(model)
    # each q/k/v/o: r * (in + out) = 4 * (16 + 16); 8 wrapped layers
    assert trainable == 8 * r * (16 + 16)
    assert trainable < total
    for n, p in model.named_parameters():
        assert p.requires_grad == ("lora_" in n)


def test_scaling_is_alpha_over_r():
    layer = LoRALinear(nn.Linear(8, 8), r=4, alpha=16)
    assert layer.scaling == 4.0


def test_gradients_only_reach_adapters_and_A_grad_is_zero_at_init(base_and_x):
    base, x = base_and_x
    model = copy.deepcopy(base)
    inject_lora(model, ATTN_TARGETS, r=4, alpha=8)
    model(x).sum().backward()
    layer = model.layers[0].q_proj
    assert layer.base.weight.grad is None            # frozen
    assert model.lm_head.weight.grad is None         # frozen
    # dL/dA is proportional to B, and B == 0 at init -> A gets no gradient yet,
    assert torch.count_nonzero(layer.lora_A.grad) == 0
    # while B does receive a gradient (this is what "kick-starts" training).
    assert torch.count_nonzero(layer.lora_B.grad) > 0


def test_merge_is_equivalent_and_reversible(base_and_x):
    base, x = base_and_x
    model = copy.deepcopy(base)
    inject_lora(model, ALL_TARGETS, r=4, alpha=8)
    randomize_B(model)
    before = model(x)
    assert not torch.allclose(before, base(x), atol=1e-4)  # adapters do something

    merge_lora(model)
    assert torch.allclose(model(x), before, atol=1e-5)
    unmerge_lora(model)
    assert torch.allclose(model(x), before, atol=1e-5)


def test_merge_and_unload_yields_plain_model(base_and_x):
    base, x = base_and_x
    model = copy.deepcopy(base)
    inject_lora(model, ALL_TARGETS, r=4, alpha=8)
    randomize_B(model)
    before = model(x)
    merge_and_unload(model)
    assert not lora_modules(model)
    assert all(isinstance(m, nn.Linear) for n, m in model.named_modules() if n.endswith("proj"))
    assert torch.allclose(model(x), before, atol=1e-5)


def test_save_and_load_roundtrip(tmp_path, base_and_x):
    base, x = base_and_x
    model = copy.deepcopy(base)
    cfg = dict(target_modules=ATTN_TARGETS, r=4, alpha=8, dropout=0.0)
    inject_lora(model, **cfg)
    randomize_B(model)
    out = model(x)
    path = tmp_path / "adapter.pt"
    save_lora(model, str(path), cfg)

    fresh = copy.deepcopy(base)
    load_lora(fresh, str(path))
    assert torch.allclose(fresh(x), out, atol=1e-6)


def test_bf16_base_with_fp32_adapters_runs():
    torch.manual_seed(0)
    lin = nn.Linear(16, 16).to(torch.bfloat16)
    layer = LoRALinear(lin, r=4, alpha=8)
    nn.init.normal_(layer.lora_B, std=0.1)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    y = layer(x)
    assert y.dtype == torch.bfloat16
    assert layer.lora_A.dtype == torch.float32
