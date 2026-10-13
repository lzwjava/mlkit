"""LoRA — Low-Rank Adaptation (Hu et al., 2021) from scratch.

Freezes the pretrained weights and injects trainable low-rank residuals
``delta_W = (alpha/r) * B @ A`` into chosen linear layers (default: the
attention QKV projection).  Only ``A`` and ``B`` are optimised, so a
fine-tune updates ~1% of the parameters.
"""

from __future__ import annotations

import numpy as np

from ..dl import nn
from ..dl.tensor import Tensor
from ..dl.optim import get_optimizer, clip_grad_norm_
from .model import TinyGPT, train_lm


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * (x @ A @ B);  A: (d_in, r), B: (r, d_out)."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0):
        self.base = base
        d_in, d_out = base.weight.shape
        self.rank = rank
        self.scaling = alpha / rank
        std = 1.0 / max(np.sqrt(d_in), 1.0)
        self.A = Tensor(np.random.normal(0, std, size=(d_in, rank)),
                        requires_grad=True, name="lora_A")
        self.B = Tensor(np.zeros((rank, d_out), dtype=np.float32),
                        requires_grad=True, name="lora_B")
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
        return self.base(x) + (x @ self.A @ self.B) * self.scaling

    def merge_(self):
        """Fold the adapter into the base weight for fast inference."""
        if not self.merged:
            self.base.weight.data += self.scaling * (self.A.data @ self.B.data)
            self.merged = True
        return self.base


# --------------------------------------------------------------------------

def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def apply_lora(model: TinyGPT, targets=("c_attn",), rank: int = 4,
               alpha: float = 1.0, seed: int = 0) -> list[str]:
    """Wrap the selected linear layers of every block with LoRALinear.

    Idempotent: layers already carrying a matching-rank adapter are kept.
    """
    np.random.seed(seed)
    freeze(model)
    wrapped = []
    for i, blk in enumerate(model.blocks):
        for attr in targets:
            base = getattr(blk.attn, attr, None)
            if isinstance(base, LoRALinear):
                if base.rank == rank:
                    wrapped.append(f"blocks.{i}.attn.{attr}")
                    continue
                base = base.base                    # re-wrap at new rank
            if isinstance(base, nn.Linear):
                setattr(blk.attn, attr, LoRALinear(base, rank=rank, alpha=alpha))
                wrapped.append(f"blocks.{i}.attn.{attr}")
    if not wrapped:
        raise ValueError(f"no layers matched targets={targets}")
    return wrapped


def lora_parameters(model: TinyGPT) -> list[Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def lora_state(model: TinyGPT) -> dict:
    """Only the adapter tensors (names ending in ``.A`` / ``.B``)."""
    return {k: v.data.copy() for k, v in model.named_parameters()
            if k.endswith(".A") or k.endswith(".B")}


def save_adapter(model: TinyGPT, path: str):
    state = {k: v for k, v in lora_state(model).items()}
    if not state:
        raise ValueError("model has no LoRA adapters")
    np.savez(path, **state)
    return path


def load_adapter(model: TinyGPT, path: str):
    data = np.load(path)
    own = dict(model.named_parameters())
    loaded = 0
    for k in data.files:
        if k in own:
            own[k].data[...] = data[k]
            loaded += 1
    if loaded == 0:
        raise KeyError("no adapter keys matched this model")
    return loaded


def merge_lora(model: TinyGPT) -> int:
    n = 0
    for blk in model.blocks:
        for attr in ("c_attn", "c_proj"):
            mod = getattr(blk.attn, attr, None)
            if isinstance(mod, LoRALinear):
                mod.merge_()
                setattr(blk.attn, attr, mod.base)
                n += 1
    return n


# --------------------------------------------------------------------------

def lora_finetune(model: TinyGPT, tokens, tokenizer, *, rank: int = 4,
                  alpha: float = 1.0, targets=("c_attn",), steps: int = 300,
                  batch_size: int = 16, lr: float = 3e-3,
                  objective: str = "clm", seed: int = 0, clip: float = 1.0,
                  log_every: int = 50, verbose: bool = True) -> dict:
    """Fine-tune a (frozen) model with LoRA adapters on `tokens`."""
    wrapped = apply_lora(model, targets=targets, rank=rank, alpha=alpha,
                         seed=seed)
    params = lora_parameters(model)
    total = model.num_params(trainable_only=False)
    trainable = model.num_params(trainable_only=True)
    if verbose:
        print(f"    adapters on {len(wrapped)} layers ({wrapped[0]} ...)")
        print(f"    trainable {trainable:,} / {total:,} params "
              f"({100 * trainable / max(total, 1):.1f}%)  rank={rank} "
              f"alpha={alpha}")

    hist = train_lm(model, tokens, objective=objective,
                    mask_id=tokenizer.mask_id if objective == "mlm" else None,
                    steps=steps, batch_size=batch_size, lr=lr,
                    optimizer="adam", weight_decay=0.0, clip=clip,
                    log_every=log_every, verbose=verbose, seed=seed,
                    lora_params=params)
    hist["trainable_params"] = trainable
    hist["total_params"] = total
    return hist