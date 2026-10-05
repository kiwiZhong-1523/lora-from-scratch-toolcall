"""A from-scratch LoRA implementation in plain PyTorch (no `peft`).

Idea (Hu et al., 2021): freeze the pretrained weight W (out x in) and learn a
low-rank update  dW = (alpha / r) * B @ A,  with A: (r x in), B: (out x r).

    h = W x + (alpha / r) * B (A x)

- A is initialised randomly (Kaiming uniform, same as nn.Linear), B with zeros,
  so dW = 0 at step 0 and the adapted model starts exactly equal to the base.
- Only A and B are trained -> r * (in + out) trainable params per wrapped layer.
- After training, dW can be folded into W (`merge`), so inference has zero
  extra latency and the model can be exported / quantized like a normal one.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

# Module-name presets for Qwen2.5 / Llama-style decoder blocks.
ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
ALL_TARGETS = ATTN_TARGETS + MLP_TARGETS


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear and adds a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("rank r must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.merged = False

        in_f, out_f = base.in_features, base.out_features
        # Adapter weights are kept in fp32 even if the base is bf16/fp16:
        # tiny tensors, and it keeps optimizer updates numerically stable.
        self.lora_A = nn.Parameter(torch.empty(r, in_f, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def delta_weight(self) -> torch.Tensor:
        """The full-rank update (alpha/r) * B @ A, shape (out, in), fp32."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        update = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.t()
        update = (update @ self.lora_B.t()) * self.scaling
        return out + update.to(out.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        w = self.base.weight
        w.add_(self.delta_weight().to(w.dtype))
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        w = self.base.weight
        w.sub_(self.delta_weight().to(w.dtype))
        self.merged = False

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4g}, merged={self.merged}"


def inject_lora(
    model: nn.Module,
    target_modules: Iterable[str] = ATTN_TARGETS,
    r: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
) -> int:
    """Freeze the whole model, then wrap every nn.Linear whose *last* name
    component is in `target_modules` with a LoRALinear. Returns #layers wrapped."""
    targets = set(target_modules)
    for p in model.parameters():
        p.requires_grad_(False)

    to_wrap = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            to_wrap.append(name)

    for name in to_wrap:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
    return len(to_wrap)


def lora_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def merge_lora(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for _, m in lora_modules(model):
        m.unmerge()


@torch.no_grad()
def merge_and_unload(model: nn.Module) -> nn.Module:
    """Fold every adapter into its base weight and replace LoRALinear by the
    plain nn.Linear. The result is a vanilla model ready for export/quantization."""
    for name, m in lora_modules(model):
        m.merge()
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, m.base)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Returns (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors (a few MB instead of the whole model)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def save_lora(model: nn.Module, path: str, config: dict) -> None:
    """`config` must contain target_modules, r, alpha (and optionally dropout)."""
    torch.save({"config": config, "state": lora_state_dict(model)}, path)


def load_lora(model: nn.Module, path: str) -> dict:
    """Inject adapters according to the saved config, then load their weights."""
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    inject_lora(
        model,
        target_modules=cfg["target_modules"],
        r=cfg["r"],
        alpha=cfg["alpha"],
        dropout=cfg.get("dropout", 0.0),
    )
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected LoRA keys: {unexpected[:5]}")
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        raise RuntimeError(f"missing LoRA keys: {lora_missing[:5]}")
    return cfg
