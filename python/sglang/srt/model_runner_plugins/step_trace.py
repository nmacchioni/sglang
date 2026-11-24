"""Per-forward trace at the worker boundary, for tests and step-overhead runs.

Off unless ``SGLANG_MODEL_RUNNER_STEP_TRACE_DIR`` is set. Each record says which
path ran (direct, or the runner's steps), the step's shape, whether it replayed
a graph (and, on the step path, ``Step.graph``), and the host time spent at the
boundary; optionally the inputs and raw logits before sampling.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from typing import TYPE_CHECKING, Optional, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_runner_plugins.api import GraphChoice, KVTables


def _slot_row_pool(pool):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    if isinstance(pool, MHATokenToKVPool) and not pool.use_hnd:
        return pool
    return None


class AcceptTrace:
    """Speculative acceptance per verify, beside the step trace: one JSON line
    per accept (``accepts-<pid>.jsonl``), on the direct and the step path."""

    def __init__(self, *, directory: str):
        self._path = os.path.join(directory, f"accepts-{os.getpid()}.jsonl")
        self._count = 0

    def record_accept(self, batch, accept) -> None:
        """``accept``: the tokens, accept lengths and accept indices (with the
        bonus token) SGLang took from one verify of ``batch``. Syncs the host."""
        record = {
            "index": self._count,
            "rids": [req.rid for req in batch.reqs],
            "seq_lens": batch.seq_lens.tolist(),
            "accept_lens": accept.accept_lens.tolist(),
            "accept_index": accept.accept_index.tolist(),
            "predict": accept.tokens.tolist(),
        }
        self._count += 1
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")


# The process's accept trace: None unless the step trace is on. Speculative
# workers check it after each accept.
ACCEPT_TRACE: Optional[AcceptTrace] = None


class StepTrace:
    @classmethod
    def maybe_create(cls) -> Optional[StepTrace]:
        global ACCEPT_TRACE
        directory = envs.SGLANG_MODEL_RUNNER_STEP_TRACE_DIR.get()
        if directory is None:
            return None
        if ACCEPT_TRACE is None:
            os.makedirs(directory, exist_ok=True)
            ACCEPT_TRACE = AcceptTrace(directory=directory)
        return cls(
            directory=directory,
            save_logits=envs.SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS.get(),
        )

    def __init__(self, *, directory: str, save_logits: bool):
        os.makedirs(directory, exist_ok=True)
        parallel = get_parallel()
        # Which rank wrote the record: each rank traces to its own file.
        self._ranks = {
            "tp_rank": parallel.tp_rank,
            "pp_rank": parallel.pp_rank,
            "attn_dp_rank": parallel.attn_dp_rank,
            "attn_cp_rank": parallel.attn_cp_rank,
            "moe_ep_rank": parallel.moe_ep_rank,
        }
        # Under expert parallelism: which expert placement each forward ran
        # with, so a rebalance (EPLB) shows up between two records.
        self._expert_layout = parallel.moe_ep_size > 1
        self._prefill_cp = parallel.attn_cp_size > 1
        self._prefix = os.path.join(directory, f"steps-{os.getpid()}")
        self._save_logits = save_logits
        self._count = 0
        # The KV pool whose bytes prefix digests read: bound on the step path.
        self._kv_pool = None

    def bind_kv_pool(self, pool) -> None:
        """Digest prefix KV bytes from ``pool`` when it keeps slot-row K/V
        buffers per layer (an MHA pool, or the view of a runner's KV)."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if isinstance(pool, MHATokenToKVPool) and not pool.use_hnd:
            self._kv_pool = pool

    def start(self) -> float:
        return time.perf_counter()

    def record_draft(
        self,
        *,
        tic: float,
        runner: str,
        execute_seconds: float,
        step,
        forward_batch: ForwardBatch,
        out,
    ) -> None:
        """One DRAFT or DRAFT_EXTEND Step of a built-in draft model."""
        graph = step.graph
        record = {
            "index": self._count,
            **self._ranks,
            "path": "step",
            "runner": runner,
            "role": "draft",
            "mode": step.mode.value,
            "batch_size": step.rows.batch_size,
            "num_tokens": int(step.input_ids.shape[0]),
            "host_us": (time.perf_counter() - tic) * 1e6,
            "execute_us": execute_seconds * 1e6,
            "replayed_graph": not graph.is_eager,
            "step_graph": None if graph.is_eager else [graph.phase.value, graph.bucket],
            "draft_tokens": None,
        }
        if out.draft is not None and self._save_logits:
            record["draft_tokens"] = out.draft.tokens.tolist()
        with open(f"{self._prefix}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        self._count += 1

    def record(
        self,
        *,
        tic: float,
        runner: Optional[str],
        execute_seconds: Optional[float],
        forward_batch: ForwardBatch,
        logits_output: Union[LogitsProcessorOutput, PPProxyTensors],
        replayed_graph: bool,
        graph: Optional[GraphChoice] = None,
        kv: Optional[KVTables] = None,
        model_runner=None,
        adapters: Optional[torch.Tensor] = None,
        sampling_owner: Optional[str] = None,
        plan_runs: Optional[int] = None,
    ) -> None:
        host_seconds = time.perf_counter() - tic
        # A pipeline stage before the last hands the next stage proxy tensors.
        pp_out = (
            logits_output.tensors if isinstance(logits_output, PPProxyTensors) else None
        )
        pooled = _pooled(logits_output)
        if pp_out is not None:
            outputs = {f"pp_out.{name}": t for name, t in sorted(pp_out.items())}
        elif pooled is not None:
            outputs = {"pooled": pooled}
        elif logits_output.next_token_logits is None:
            # A denoising pass: a logit row per input row.
            outputs = {"logits": logits_output.full_logits}
        else:
            outputs = {"logits": logits_output.next_token_logits}
        record = {
            "index": self._count,
            **self._ranks,
            "path": "direct" if runner is None else "step",
            "runner": runner,
            "mode": forward_batch.forward_mode.name.lower(),
            "batch_size": forward_batch.batch_size,
            "num_tokens": int(forward_batch.input_ids.shape[0]),
            "prefix_lens": forward_batch.extend_prefix_lens_cpu,
            "host_us": host_seconds * 1e6,
            "execute_us": None if execute_seconds is None else execute_seconds * 1e6,
            "prompt_logprobs": pp_out is None
            and pooled is None
            and logits_output.input_token_logprobs is not None,
            "pooled": pooled is not None,
            "pp_out": None if pp_out is None else sorted(pp_out),
            # Input rows per DP rank, as gathered at the MLP-sync rendezvous.
            "global_num_tokens": forward_batch.original_global_num_tokens_cpu,
            "replayed_graph": replayed_graph,
            # Step path only: [phase, bucket], or None when Step.graph is eager.
            "step_graph": (
                None
                if graph is None or graph.is_eager
                else [graph.phase.value, graph.bucket]
            ),
            # Two-batch overlap: this rank's split, and whether the eager
            # forward built the two micro-batches (graph replays split inside).
            "tbo_split_seq_index": forward_batch.tbo_split_seq_index,
            "tbo_children": forward_batch.tbo_children is not None,
            # Prefill context parallelism sharded this extend across the CP
            # group (the forward's own test, on the batch it ran).
            "cp_sharded": self._cp_sharded(forward_batch),
            # Under DP attention: the padding the forward applied, and the
            # per-rank rows after it.
            "dp_padding": (
                None
                if forward_batch.dp_padding_mode is None
                else forward_batch.dp_padding_mode.name
            ),
            "padded_num_tokens": forward_batch.global_num_tokens_cpu,
            "expert_layout": self._expert_layout_crc(),
            # LoRA: each request's adapter, and (step path) its Step.adapters slot.
            "lora_ids": (
                None
                if not forward_batch.lora_ids
                or all(uid is None for uid in forward_batch.lora_ids)
                else list(forward_batch.lora_ids[: forward_batch.batch_size])
            ),
            "adapters": None if adapters is None else adapters.tolist(),
            # Step path only: who sampled the step ("sglang" or "plugin").
            "sampling_owner": sampling_owner,
            # Step path only: the executions of this step's plan so far (a
            # denoising pass re-executes its batch's plan).
            "plan_runs": plan_runs,
            # Multimodal items the step's requests carry.
            "mm_items": sum(
                len(mm.mm_items)
                for mm in forward_batch.mm_inputs or ()
                if mm is not None
            ),
            # Step path only: the window and state tables the Step carried.
            **_sub_pool_summary(kv),
            # Step path, with saved tensors: per request with a cached prefix
            # on an extend step, a CRC of that prefix's K and V bytes.
            "prefix_kv_crc": self._prefix_kv_crcs(forward_batch, kv, model_runner),
        }
        with open(f"{self._prefix}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        if self._save_logits and all(t is not None for t in outputs.values()):
            inputs = {
                "input_ids": forward_batch.input_ids,
                "positions": forward_batch.positions,
                "out_cache_loc": forward_batch.out_cache_loc,
                "seq_lens": forward_batch.seq_lens,
                "req_pool_indices": forward_batch.req_pool_indices,
                **_sub_pool_tensors(kv, forward_batch),
            }
            torch.save(
                {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in {**inputs, **outputs}.items()
                },
                f"{self._prefix}-{self._count:05d}.pt",
            )
        self._count += 1

    def _prefix_kv_crcs(
        self, forward_batch: ForwardBatch, kv: Optional[KVTables], model_runner
    ) -> Optional[list[int]]:
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        if not self._save_logits or not prefix_lens or not any(prefix_lens):
            return None
        # The direct path reads the worker's own pool and page table.
        if kv is not None:
            pool, page_table = self._kv_pool, kv.page_table
        elif model_runner is not None:
            pool = _slot_row_pool(model_runner.token_to_kv_pool)
            page_table = model_runner.req_to_token_pool.req_to_token
        else:
            pool = None
        if pool is None:
            return None
        crcs = []
        for row, prefix in zip(forward_batch.req_pool_indices.tolist(), prefix_lens):
            if not prefix:
                continue
            slots = page_table[row, :prefix].long()
            crc = 0
            for buffer in (*pool.k_buffer, *pool.v_buffer):
                rows = buffer[slots].contiguous().view(torch.uint8)
                crc = zlib.crc32(rows.cpu().numpy().tobytes(), crc)
            crcs.append(crc)
        return crcs

    def _cp_sharded(self, forward_batch: ForwardBatch) -> bool:
        if not self._prefill_cp:
            return False
        from sglang.srt.layers.cp.utils import is_cp_active

        return is_cp_active(forward_batch)

    def _expert_layout_crc(self) -> Optional[int]:
        if not self._expert_layout:
            return None
        from sglang.srt.eplb.expert_location import get_global_expert_location_metadata

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return None
        return zlib.crc32(metadata.physical_to_logical_map_cpu.numpy().tobytes())


def _sub_pool_summary(kv: Optional[KVTables]) -> dict:
    """Which sub-pool tables a step carried, and how many state operations."""
    if kv is None:
        return {"kv_window": None, "kv_state": None}
    state = kv.state
    return {
        "kv_window": kv.window is not None,
        "kv_state": None
        if state is None
        else {
            "clear": _rows(state.clear),
            "copy": _rows(state.copy_src),
            "checkpoints": (
                0
                if state.checkpoint_mask is None
                else int(state.checkpoint_mask.sum().item())
            ),
        },
    }


def _rows(tensor: Optional[torch.Tensor]) -> int:
    return 0 if tensor is None else int(tensor.shape[0])


def _sub_pool_tensors(kv: Optional[KVTables], forward_batch: ForwardBatch) -> dict:
    """The window write slots, and each request's state slot, as the Step named them."""
    if kv is None:
        return {}
    tensors = {}
    if kv.window is not None:
        tensors["window_write_slots"] = kv.window.write_slots
    if kv.state is not None:
        tensors["state_slots"] = kv.state.req_to_state[forward_batch.req_pool_indices]
        if kv.state.copy_src is not None:
            tensors["state_copy_src"] = kv.state.copy_src
            tensors["state_copy_dst"] = kv.state.copy_dst
    return tensors


def _pooled(logits_output) -> Optional[torch.Tensor]:
    """A pooling model's outputs, as one tensor (per-request ones concatenated)."""
    from sglang.srt.layers.pooler import EmbeddingPoolerOutput

    if not isinstance(logits_output, EmbeddingPoolerOutput):
        return None
    embeddings = logits_output.embeddings
    if isinstance(embeddings, torch.Tensor):
        return embeddings
    return torch.cat([t.reshape(-1) for t in embeddings])
