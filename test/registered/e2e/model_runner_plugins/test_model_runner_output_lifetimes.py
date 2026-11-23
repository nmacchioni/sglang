"""Output lifetimes on the device: back-to-back graph replays while the previous
step's output is still being read.

The reference runner's decode graphs write logits into one shared buffer. SGLang
releases a step's output as soon as its last consumer is enqueued, so the next
replay is launched while a host copy of the previous logits may still be
queued on the copy stream. The copy here is held back on purpose; the next step
must not overwrite the buffer until the release event fires. A runner that
ignores the event is run as a control and must corrupt the copy, which shows
the fixture detects a buffer reused before its consumer finished.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_runner_plugins.api import (
    GraphBuckets,
    GraphChoice,
    GraphPhase,
    OutputBuffer,
)
from sglang.srt.model_runner_plugins.step_forward import StepForward
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference.model import DecoderConfig, DenseDecoder
from sglang.test.model_runner_reference.plugin import ReferenceRunner
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

_CONFIG = DecoderConfig(
    vocab_size=512,
    hidden_size=64,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=128,
    dtype=torch.bfloat16,
)
_SLOTS = 64
# GPU cycles the held-back copy waits (tens of ms); the next replay of this
# tiny model takes well under that.
_HOLD_COPY_CYCLES = 200_000_000


def _weights(device):
    c = _CONFIG
    g = torch.Generator(device="cpu").manual_seed(0)

    def rand(*shape):
        return (torch.randn(*shape, generator=g) * 0.2).to(device)

    w = {
        "model.embed_tokens.weight": rand(c.vocab_size, c.hidden_size),
        "model.norm.weight": torch.ones(c.hidden_size, device=device),
    }
    q, kv, inter = c.num_heads * c.head_dim, c.num_kv_heads * c.head_dim, 128
    for layer in range(c.num_layers):
        p = f"model.layers.{layer}."
        w[p + "input_layernorm.weight"] = torch.ones(c.hidden_size, device=device)
        w[p + "post_attention_layernorm.weight"] = torch.ones(
            c.hidden_size, device=device
        )
        w[p + "self_attn.q_proj.weight"] = rand(q, c.hidden_size)
        w[p + "self_attn.k_proj.weight"] = rand(kv, c.hidden_size)
        w[p + "self_attn.v_proj.weight"] = rand(kv, c.hidden_size)
        w[p + "self_attn.o_proj.weight"] = rand(c.hidden_size, q)
        w[p + "mlp.gate_proj.weight"] = rand(inter, c.hidden_size)
        w[p + "mlp.up_proj.weight"] = rand(inter, c.hidden_size)
        w[p + "mlp.down_proj.weight"] = rand(c.hidden_size, inter)
    return w


def _runner(device) -> ReferenceRunner:
    runner = ReferenceRunner()
    model = DenseDecoder(config=_CONFIG, weights=_weights(device), device=device)
    model.allocate_kv(_SLOTS)
    runner._model = model
    runner.capture(GraphBuckets(decode=(1, 2)))
    return runner


def _decode_batch(*, token: int, position: int, device) -> ForwardBatch:
    """Request 1 decodes ``token`` at ``position``; its KV holds positions
    0..position, slot i for position i."""
    seq_len = position + 1
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([token], device=device),
        req_pool_indices=torch.tensor([1], device=device),
        seq_lens=torch.tensor([seq_len], device=device),
        seq_lens_cpu=torch.tensor([seq_len]),
        out_cache_loc=torch.tensor([position + 8], device=device),
        seq_lens_sum=seq_len,
        positions=torch.tensor([position], device=device),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


class TestBackToBackReplay(CustomTestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        page_table = torch.zeros(4, 32, dtype=torch.int64, device=self.device)
        page_table[1] = torch.arange(32, device=self.device) + 8
        self.page_table = page_table

    def _steps(self, runner):
        return StepForward(
            runner_name="reference",
            executor=runner,
            page_table=self.page_table,
            vocab_size=_CONFIG.vocab_size,
            native_prompt_logprobs=False,
            overlap=True,
            shared_outputs=(OutputBuffer.LOGITS,),
            choose_graph=lambda batch: GraphChoice(phase=GraphPhase.DECODE, bucket=1),
        )

    def _two_steps(self, runner, *, hold_copy: bool):
        """Step 0's logits are copied on a side stream, held back when
        ``hold_copy``; step 1 is launched right after step 0's release."""
        steps = self._steps(runner)
        copy_stream = torch.cuda.Stream()
        scheduler = SimpleNamespace(
            device_module=torch.cuda,
            tp_worker=SimpleNamespace(step_forward=steps),
        )
        host = torch.empty(1, _CONFIG.vocab_size, pin_memory=True)
        out0 = steps.forward(_decode_batch(token=3, position=0, device=self.device))
        logits0 = out0.logits_output.next_token_logits
        self.assertTrue(runner._graphs.holds(logits0))
        copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(copy_stream):
            if hold_copy:
                torch.cuda._sleep(_HOLD_COPY_CYCLES)
            host.copy_(logits0, non_blocking=True)
        result = SimpleNamespace(
            step_output_lease=steps.take_lease(), logits_output=out0.logits_output
        )
        Scheduler._release_step_outputs(scheduler, result, consumer_stream=copy_stream)
        out1 = steps.forward(_decode_batch(token=7, position=1, device=self.device))
        logits1 = out1.logits_output.next_token_logits.clone()
        torch.cuda.synchronize()
        steps.close()
        return host.clone(), logits1.cpu()

    def test_a_held_back_consumer_reads_its_own_step(self):
        expected0, expected1 = self._two_steps(_runner(self.device), hold_copy=False)
        self.assertFalse(torch.equal(expected0, expected1))

        copied0, logits1 = self._two_steps(_runner(self.device), hold_copy=True)
        self.assertTrue(torch.equal(copied0, expected0))
        self.assertTrue(torch.equal(logits1, expected1))

    def test_a_runner_ignoring_the_release_event_is_caught(self):
        expected0, expected1 = self._two_steps(_runner(self.device), hold_copy=False)
        careless = _runner(self.device)
        careless._wait_logits_free = lambda: None
        copied0, _ = self._two_steps(careless, hold_copy=True)
        # The copy read step 1's logits: the buffer was reused under it.
        self.assertTrue(torch.equal(copied0, expected1))


if __name__ == "__main__":
    unittest.main(verbosity=3)
