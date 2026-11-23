"""A dense Llama-family decoder (Llama, Qwen2, Qwen3) in plain PyTorch.

Unfused on purpose: it shares no kernel with SGLang's models, so a match against
SGLang's own runner checks the plugin contract, not shared code. Each layer is
split at attention, so decode graphs (``graphs.py``) can capture the rest.
K and V live in one paged tensor per layer, indexed by SGLang's physical slots.

Tensor parallel over the TP group SGLang hands to ``load``: Q, K, V, gate and up
projections are column-parallel (whole heads and intermediate columns per
rank), O and down projections row-parallel with an all-reduce, and the LM head
vocabulary-parallel. Embeddings and norms are replicated.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Optional

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.model_runner_plugins.api import SINGLE_RANK, RankGroup, Step, StepMode


class DecoderConfig(msgspec.Struct, frozen=True, kw_only=True):
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    dtype: torch.dtype

    @classmethod
    def from_hf(cls, hf: dict) -> DecoderConfig:
        if hf.get("rope_scaling") is not None:
            raise ValueError("the reference runner implements unscaled RoPE only")
        return cls(
            vocab_size=hf["vocab_size"],
            hidden_size=hf["hidden_size"],
            num_layers=hf["num_hidden_layers"],
            num_heads=hf["num_attention_heads"],
            num_kv_heads=hf.get("num_key_value_heads", hf["num_attention_heads"]),
            head_dim=hf.get("head_dim")
            or hf["hidden_size"] // hf["num_attention_heads"],
            rms_norm_eps=hf["rms_norm_eps"],
            rope_theta=hf.get("rope_theta", 10000.0),
            max_position_embeddings=hf["max_position_embeddings"],
            dtype={"float16": torch.float16, "float32": torch.float32}.get(
                hf.get("torch_dtype"), torch.bfloat16
            ),
        )


def resolve_checkpoint(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path, allow_patterns=["*.json", "*.safetensors"])


def load_weights(path: str, *, device: torch.device) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    weights = {}
    for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        weights.update(load_file(file, device=str(device)))
    if not weights:
        raise FileNotFoundError(f"no safetensors weights under {path}")
    return weights


def load_config(path: str) -> DecoderConfig:
    with open(os.path.join(path, "config.json")) as f:
        return DecoderConfig.from_hf(json.load(f))


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # In FP32, as SGLang's rotary kernels compute it. Measured on Qwen3-0.6B:
    # bf16 RoPE (as HF applies it) put forced-token logprobs up to 0.30 nats
    # from SGLang's own runner; FP32 RoPE brings that to 0.17.
    x32 = x.float()
    return (x32 * cos + _rotate_half(x32) * sin).to(x.dtype)


class PagedKV(msgspec.Struct, kw_only=True):
    # [num_layers, rows, num_kv_heads, head_dim]; row = physical slot.
    k: torch.Tensor
    v: torch.Tensor


# Sharded along dim 0 (output features) or dim 1 (input features) per rank.
_COLUMN_PARALLEL = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
_ROW_PARALLEL = ("o_proj", "down_proj")


def shard_config(config: DecoderConfig, tp: RankGroup) -> DecoderConfig:
    """The per-rank shapes under ``tp``: whole heads per rank."""
    if config.num_heads % tp.size or config.num_kv_heads % tp.size:
        raise ValueError(
            f"the reference runner splits whole heads: {config.num_heads} query "
            f"and {config.num_kv_heads} KV heads over {tp.size} ranks"
        )
    return msgspec.structs.replace(
        config,
        num_heads=config.num_heads // tp.size,
        num_kv_heads=config.num_kv_heads // tp.size,
    )


def _shard(name: str, t: torch.Tensor, tp: RankGroup) -> torch.Tensor:
    """This rank's slice of checkpoint tensor ``name``, in its own storage."""
    module = name.rsplit(".", 2)[-2] if name.count(".") >= 2 else ""
    if tp.size == 1:
        return t
    if module in _COLUMN_PARALLEL:
        return t.chunk(tp.size, dim=0)[tp.rank].clone()
    if module in _ROW_PARALLEL:
        if name.endswith(".bias"):
            # Added once, after the all-reduce sums the partial products.
            return t if tp.rank == 0 else torch.zeros_like(t)
        return t.chunk(tp.size, dim=1)[tp.rank].clone()
    return t


class DenseDecoder:
    def __init__(
        self,
        *,
        config: DecoderConfig,
        weights: dict[str, torch.Tensor],
        device: torch.device,
        tp: RankGroup = SINGLE_RANK,
    ):
        # ``full_config`` is the checkpoint's; ``config`` this rank's slice.
        self.full_config = config
        self.config = shard_config(config, tp)
        self.tp = tp
        self.device = device
        if "lm_head.weight" not in weights:
            weights["lm_head.weight"] = weights["model.embed_tokens.weight"]
        self.w = {
            name: _shard(name, t, tp).to(config.dtype).contiguous()
            for name, t in weights.items()
            if name != "lm_head.weight"
        }
        self.w["lm_head.weight"] = self._vocab_shard(weights["lm_head.weight"])
        config = self.config
        half = config.head_dim // 2
        self._inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, half, device=device).float() * 2 / config.head_dim)
        )
        self.kv: Optional[PagedKV] = None

    def bytes_per_slot(self) -> int:
        c = self.config
        return 2 * c.num_layers * c.num_kv_heads * c.head_dim * c.dtype.itemsize

    def allocate_kv(self, rows: int) -> PagedKV:
        c = self.config
        shape = (c.num_layers, rows, c.num_kv_heads, c.head_dim)
        self.kv = PagedKV(
            k=torch.zeros(shape, dtype=c.dtype, device=self.device),
            v=torch.zeros(shape, dtype=c.dtype, device=self.device),
        )
        return self.kv

    def hidden_states(self, step: Step) -> torch.Tensor:
        """[T, hidden_size] final-norm hidden states of the step's input rows."""
        x = self.embed(step.input_ids)
        cos, sin = self.rope(step.positions)
        spans = request_spans(step)
        for layer in range(self.config.num_layers):
            q = self.pre_attention(
                x, layer=layer, cos=cos, sin=sin, write_slots=step.kv.write_slots
            )
            attn = self.attention(q, layer=layer, step=step, spans=spans)
            x = self.post_attention(x, attn, layer=layer)
        return self.final_norm(x)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.w["model.embed_tokens.weight"][input_ids]

    def final_norm(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.w["model.norm.weight"], self.config.rms_norm_eps)

    def pre_attention(
        self,
        x: torch.Tensor,
        *,
        layer: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        write_slots: torch.Tensor,
    ) -> torch.Tensor:
        """Writes the layer's K and V of every row to its slot; returns Q."""
        c = self.config
        p = f"model.layers.{layer}."
        h = _rms_norm(x, self.w[p + "input_layernorm.weight"], c.rms_norm_eps)
        q = self._linear(h, p + "self_attn.q_proj").view(-1, c.num_heads, c.head_dim)
        k = self._linear(h, p + "self_attn.k_proj").view(-1, c.num_kv_heads, c.head_dim)
        v = self._linear(h, p + "self_attn.v_proj").view(-1, c.num_kv_heads, c.head_dim)
        if p + "self_attn.q_norm.weight" in self.w:
            q = _rms_norm(q, self.w[p + "self_attn.q_norm.weight"], c.rms_norm_eps)
            k = _rms_norm(k, self.w[p + "self_attn.k_norm.weight"], c.rms_norm_eps)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        self.kv.k[layer, write_slots] = k
        self.kv.v[layer, write_slots] = v
        return q

    def post_attention(
        self, x: torch.Tensor, attn: torch.Tensor, *, layer: int
    ) -> torch.Tensor:
        c = self.config
        p = f"model.layers.{layer}."
        o = self._linear(attn.reshape(attn.shape[0], -1), p + "self_attn.o_proj")
        x = x + self._all_reduce(o)
        h = _rms_norm(x, self.w[p + "post_attention_layernorm.weight"], c.rms_norm_eps)
        gate = self._linear(h, p + "mlp.gate_proj")
        up = self._linear(h, p + "mlp.up_proj")
        down = self._linear(F.silu(gate) * up, p + "mlp.down_proj")
        return x + self._all_reduce(down)

    def logits_shard(self, hidden: torch.Tensor) -> torch.Tensor:
        """FP32 logits of this rank's vocabulary shard, [rows, vocab_shard]."""
        return (hidden @ self.w["lm_head.weight"].t()).float()

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """FP32 full-vocabulary logits, gathered across the TP group."""
        shard = self.logits_shard(hidden)
        if self.tp.size > 1:
            parts = [torch.empty_like(shard) for _ in range(self.tp.size)]
            torch.distributed.all_gather(parts, shard, group=self.tp.device)
            shard = torch.cat(parts, dim=-1)
        vocab = self.full_config.vocab_size
        return shard if shard.shape[-1] == vocab else shard[:, :vocab].contiguous()

    @property
    def vocab_shard_size(self) -> int:
        return self.w["lm_head.weight"].shape[0]

    def _vocab_shard(self, lm_head: torch.Tensor) -> torch.Tensor:
        """Rows [r * shard, (r + 1) * shard) of the vocabulary padded with zero
        rows to tp_size * shard."""
        vocab = lm_head.shape[0]
        shard = -(-vocab // self.tp.size)
        start = self.tp.rank * shard
        rows = lm_head[start : start + shard].to(self.full_config.dtype)
        if rows.shape[0] < shard:
            rows = torch.cat(
                [rows, rows.new_zeros(shard - rows.shape[0], rows.shape[1])]
            )
        return rows.contiguous()

    def _all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp.size > 1:
            torch.distributed.all_reduce(x, group=self.tp.device)
        return x

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(x, self.w[name + ".weight"], self.w.get(name + ".bias"))

    def rope(self, positions: torch.Tensor):
        """FP32 cos and sin, [T, 1, head_dim]."""
        freqs = positions.float()[:, None] * self._inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[:, None, :], emb.sin()[:, None, :]

    def attention(self, q: torch.Tensor, *, layer: int, step: Step, spans):
        """Causal attention of each request's query rows over its KV in the pool."""
        c = self.config
        out = torch.empty_like(q)
        groups = c.num_heads // c.num_kv_heads
        for row, (start, length, prefix, seq_len) in enumerate(spans):
            slots = step.kv.page_table[step.rows.req_pool_indices[row], :seq_len]
            keys = self.kv.k[layer, slots].repeat_interleave(groups, dim=1)
            values = self.kv.v[layer, slots].repeat_interleave(groups, dim=1)
            query = q[start : start + length]
            # Query i sits at position prefix + i and sees keys 0..prefix + i.
            mask = torch.arange(seq_len, device=q.device)[None, :] <= (
                prefix + torch.arange(length, device=q.device)[:, None]
            )
            out[start : start + length] = F.scaled_dot_product_attention(
                query.transpose(0, 1),
                keys.transpose(0, 1),
                values.transpose(0, 1),
                attn_mask=mask,
            ).transpose(0, 1)
        return out


def request_spans(step: Step) -> list[tuple[int, int, int, int]]:
    """(first input row, rows, prefix length, sequence length) per request."""
    rows = step.rows
    seq_lens = (
        rows.seq_lens_cpu if rows.seq_lens_cpu is not None else rows.seq_lens.cpu()
    ).tolist()
    if step.mode in (StepMode.DECODE, StepMode.IDLE):
        return [(b, 1, n - 1, n) for b, n in enumerate(seq_lens)]
    spans, start = [], 0
    for length, prefix, seq_len in zip(
        rows.query_lens_cpu, rows.prefix_lens_cpu, seq_lens
    ):
        spans.append((start, length, prefix, seq_len))
        start += length
    return spans


def last_rows(step: Step) -> torch.Tensor:
    """The input row of each request's last token."""
    if step.mode in (StepMode.DECODE, StepMode.IDLE):
        return torch.arange(step.rows.batch_size, device=step.input_ids.device)
    return step.rows.query_start + step.rows.query_lens - 1
