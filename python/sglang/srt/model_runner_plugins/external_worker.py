"""SGLang's worker over an external model runner: no model, no KV bytes.

``PluginOwnedModelRunner`` keeps the ``ModelRunner`` pieces SGLang still owns
for an external runner (request and slot allocators, the sampler, the runner
stream) and skips the ones the plugin owns (weights, KV bytes, attention
backends, graphs). It follows the MLX worker's stub runner; the scheduler's
remaining ``ModelRunner`` reads see the answers this module sets.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.allocation_sizing import get_req_to_token_extra_context_len
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    ModelLayerInfo,
)
from sglang.srt.model_runner_plugins.kv_view import PluginKVView, kv_view_refusal
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_runner_plugins.external import ExternalRunner

logger = logging.getLogger(__name__)

_GIB = 1 << 30
# A request needs at least this many slots of headroom; an arbitrary floor that
# keeps the default concurrency below the slot count.
_SLOTS_PER_DEFAULT_REQUEST = 2
# Arbitrary default cap, as the MLX stub runner uses.
_DEFAULT_MAX_RUNNING_REQUESTS = 4096


class _PluginOwnedKVCache(KVCache):
    """Satisfies the scheduler's KV pool reads; the plugin holds the bytes."""

    def __init__(self, *, size: int, page_size: int, dtype: torch.dtype, device):
        # KVCache.__init__ would allocate; set only what allocators and the
        # radix cache read.
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.store_dtype = dtype
        self.device = device
        self.layer_num = 0
        self.start_layer = 0
        self.end_layer = 0
        self.mem_usage = 0
        self.cpu_offloading_chunk_size = 8192
        self.layer_transfer_counter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None

    def _bytes_are_the_plugins(self):
        raise RuntimeError("The model runner plugin owns these KV bytes")

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self._bytes_are_the_plugins()

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        self._bytes_are_the_plugins()

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._bytes_are_the_plugins()

    def set_kv_buffer(self, layer, loc, cache_k, cache_v) -> None:
        self._bytes_are_the_plugins()

    def get_kv_size_bytes(self):
        return 0, 0


class _NoModel:
    """Where ModelRunner expects ``self.model``; the plugin holds the model."""

    shared_experts_config = SharedExpertsConfig()

    @staticmethod
    def forward():
        raise RuntimeError("The model runner plugin owns the model")

    @staticmethod
    def modules():
        # Speculative workers look for logits processors to configure.
        return ()


class PluginOwnedModelRunner(ModelRunner):
    canary_manager = None
    prefill_aware_swa = False

    def __init__(self, *args, runner: ExternalRunner, **kwargs):
        if get_model().is_startup_weight_load_overlap:
            raise ValueError(
                "--startup-weight-load-mode=overlap loads SGLang's own model; "
                "it does not apply to an external model runner"
            )
        self._runner = runner
        super().__init__(*args, **kwargs)

    @property
    def preloaded_weights_bytes(self) -> int:
        return 0

    @property
    def logical_max_total_num_tokens(self) -> int:
        return self.req_to_token_pool.schedulable_token_capacity(
            self.max_total_num_tokens
        )

    def initialize(self):
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=get_exec().features.enable_memory_saver
        )
        self.eplb_manager = None
        self.expert_location_updater = None
        self.init_token_oracle()
        self.sampler = create_sampler()
        self.load_model()
        num_layers = self.model_config.num_hidden_layers
        self.layer_info = ModelLayerInfo(
            start_layer=0, end_layer=num_layers, num_effective_layers=num_layers
        )
        self.kv_cache_dtype = self.dtype
        # PD peers check this string matches before they move bytes. A runner's
        # KV rows are opaque bytes per TP rank, which a transfer cannot slice
        # between TP sizes: only peers of the same runner and TP size pair.
        self.kv_cache_dtype_str = (
            f"model runner {self._runner.name} at TP {get_parallel().tp_size}"
        )
        self.uocr_greedy_graph = None
        self.decode_cuda_graph_runner = None
        self.prefill_cuda_graph_runner = None
        self.eager_runner = None
        self.attn_backend = None

    def load_model(self):
        memory = self._runner.load_result.memory
        self.model = _NoModel()
        self.sliding_window_size = None
        self.dtype = self.model_config.dtype
        self.weight_load_mem_usage = memory.weight_bytes / _GIB

    def alloc_memory_pool(self, memory_pool_config=None, *, draft_parallel_layout=None):
        """SGLang's request and slot allocators over the plugin's slots."""
        layout = self._runner.layout
        self.max_total_num_tokens = layout.num_slots
        self.max_running_requests = self._max_running_requests(layout.num_slots)
        self.req_to_token_pool = ReqToTokenPool(
            size=self.max_running_requests,
            max_context_len=(
                self.model_config.context_len + get_req_to_token_extra_context_len()
            ),
            device=self.device,
            enable_memory_saver=False,
        )
        self.token_to_kv_pool = self._kv_pool(layout)
        if layout.page_size == 1:
            self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                layout.num_slots,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=False,
            )
        else:
            self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                layout.num_slots,
                page_size=layout.page_size,
                dtype=self.kv_cache_dtype,
                device=self.device,
                kvcache=self.token_to_kv_pool,
                need_sort=False,
            )
        self._init_post_memory_pool_components()
        logger.info(
            "Model runner %r owns %d KV slots; SGLang admits up to %d requests",
            self._runner.name,
            layout.num_slots,
            self.max_running_requests,
        )

    def _kv_pool(self, layout):
        """A byte view of the runner's KV, for host copies, transfers and the
        moves of accepted drafts; a stand-in that holds no bytes when the
        layout is not viewable."""
        refusal = kv_view_refusal(layout)
        if refusal is None:
            return PluginKVView(
                layout=layout,
                device=self.device,
                move_slots=get_spec().speculative_algorithm is not None,
            )
        for in_use, feature, flag in (
            (
                get_memory().enable_hierarchical_cache,
                "the hierarchical cache",
                "drop --enable-hierarchical-cache",
            ),
            (
                get_disagg().disaggregation_mode != "null",
                "PD disaggregation",
                "serve without --disaggregation-mode",
            ),
            (
                # SGLang moves accepted drafts' KV bytes after each verify.
                get_spec().speculative_algorithm is not None,
                "speculative decoding",
                "serve without --speculative-algorithm",
            ),
        ):
            if in_use:
                raise ValueError(
                    f"--model-runner {self._runner.name} cannot use {feature}: "
                    f"its KVLayout has {refusal}; {flag}"
                )
        return _PluginOwnedKVCache(
            size=layout.num_slots,
            page_size=layout.page_size,
            dtype=self.kv_cache_dtype,
            device=self.device,
        )

    def init_attention_backends(self):
        # Attention runs inside the plugin. Positions are computed with the
        # Triton kernel, as for SGLang's default backends.
        self.attn_backend = None
        self.prefill_attention_backend_str = "triton"
        self.decode_attention_backend_str = "triton"

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        pass

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"Model runner {self._runner.name!r} runs only through its Step API"
        )

    def _max_running_requests(self, num_slots: int) -> int:
        capacity = max(1, num_slots // _SLOTS_PER_DEFAULT_REQUEST)
        requested = get_schedule().max_running_requests
        if requested is None:
            return min(capacity, _DEFAULT_MAX_RUNNING_REQUESTS)
        return min(requested, capacity)


class ExternalTpModelWorker(TpModelWorker):
    """A ``TpModelWorker`` whose forwards all run through the plugin's steps."""

    def __init__(self, *, runner: ExternalRunner, **kwargs):
        self._external_runner = runner
        super().__init__(**kwargs)

    def _init_model_runner(self):
        self._model_runner = PluginOwnedModelRunner(
            model_config=self.model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=self.gpu_id,
            nccl_port=self.nccl_port,
            server_args=self.server_args,
            runner=self._external_runner,
        )

    def get_pad_input_ids_func(self):
        return None
