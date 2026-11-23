"""Decode CUDA graphs for the reference runner, captured piecewise.

Each bucket captures one graph per span between attentions: the embedding and
layer 0's projections, then layer l's MLP with layer l+1's projections, then
the last MLP, the final norm and the LM head. Attention runs eagerly between
them over the real rows, so a graph's shapes never depend on sequence lengths.
Padded rows read token 0 at position 0 and write their K and V to the reserved
slot 0.

Logits land in one buffer shared by every bucket, as SGLang's own graph runners
share theirs, so the runner declares logits a shared output and orders each
overwrite after the previous step's release event.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from sglang.srt.model_runner_plugins.api import Step
from sglang.test.model_runner_reference.model import DenseDecoder, request_spans


class _BucketGraph:
    """The piecewise graphs of one decode bucket of ``rows`` requests."""

    def __init__(self, *, model: DenseDecoder, rows: int, logits: torch.Tensor, pool):
        c = model.config
        device = model.device
        self._model = model
        self.rows = rows
        self._input_ids = torch.zeros(rows, dtype=torch.int64, device=device)
        self._positions = torch.zeros(rows, dtype=torch.int64, device=device)
        self._write_slots = torch.zeros(rows, dtype=torch.int64, device=device)
        self._x = torch.zeros(rows, c.hidden_size, dtype=c.dtype, device=device)
        self._q = torch.zeros(
            rows, c.num_heads, c.head_dim, dtype=c.dtype, device=device
        )
        self._attn = torch.zeros_like(self._q)
        cos, sin = model.rope(self._positions)
        self._cos, self._sin = torch.zeros_like(cos), torch.zeros_like(sin)
        self._logits = logits[:rows]
        self._pieces = [
            self._capture(piece, pool)
            for piece in [self._first, *self._middles(), self._last]
        ]

    def _first(self) -> None:
        cos, sin = self._model.rope(self._positions)
        self._cos.copy_(cos)
        self._sin.copy_(sin)
        self._x.copy_(self._model.embed(self._input_ids))
        self._project(0)

    def _middles(self) -> list[Callable[[], None]]:
        def middle(layer: int) -> None:
            self._x.copy_(self._model.post_attention(self._x, self._attn, layer=layer))
            self._project(layer + 1)

        num_layers = self._model.config.num_layers
        return [lambda layer=layer: middle(layer) for layer in range(num_layers - 1)]

    def _last(self) -> None:
        layer = self._model.config.num_layers - 1
        x = self._model.post_attention(self._x, self._attn, layer=layer)
        self._logits.copy_(self._model.logits(self._model.final_norm(x)))

    def _project(self, layer: int) -> None:
        q = self._model.pre_attention(
            self._x,
            layer=layer,
            cos=self._cos,
            sin=self._sin,
            write_slots=self._write_slots,
        )
        self._q.copy_(q)

    @staticmethod
    def _capture(piece: Callable[[], None], pool) -> torch.cuda.CUDAGraph:
        # Warm up off the capture stream, as torch.cuda.graph requires.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            piece()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            piece()
        return graph

    def replay(self, step: Step, *, before_logits: Optional[Callable[[], None]]):
        bs = step.rows.batch_size
        self._input_ids.zero_()
        self._positions.zero_()
        self._write_slots.zero_()
        self._input_ids[:bs].copy_(step.input_ids)
        self._positions[:bs].copy_(step.positions)
        self._write_slots[:bs].copy_(step.kv.write_slots)
        spans = request_spans(step)
        num_layers = self._model.config.num_layers
        for layer in range(num_layers):
            self._pieces[layer].replay()
            attn = self._model.attention(
                self._q[:bs], layer=layer, step=step, spans=spans
            )
            self._attn[:bs].copy_(attn)
        if before_logits is not None:
            before_logits()
        self._pieces[num_layers].replay()
        return self._logits[:bs]


class DecodeGraphs:
    def __init__(self, *, model: DenseDecoder, buckets: tuple[int, ...]):
        self.logits = torch.zeros(
            max(buckets),
            model.config.vocab_size,
            dtype=torch.float32,
            device=model.device,
        )
        pool = torch.cuda.graph_pool_handle()
        # Largest first, so smaller buckets reuse the pool's blocks.
        self._graphs = {
            rows: _BucketGraph(model=model, rows=rows, logits=self.logits, pool=pool)
            for rows in sorted(buckets, reverse=True)
        }
        self.replays = 0

    def holds(self, tensor: torch.Tensor) -> bool:
        return tensor.untyped_storage().data_ptr() == self.logits.data_ptr()

    def replay(
        self,
        step: Step,
        *,
        bucket: int,
        before_logits: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        graph = self._graphs.get(bucket)
        if graph is None:
            raise ValueError(f"no decode graph was captured for bucket {bucket}")
        self.replays += 1
        return graph.replay(step, before_logits=before_logits)
