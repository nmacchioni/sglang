"""The reference model runner plugin, ``--model-runner reference`` once installed.

Tests install it as a pip install would, through entry point metadata:
``install_reference_runner(root)`` writes a dist-info under ``root`` and returns
the environment that puts ``root`` on a server's ``sys.path``.
"""

import glob
import json
import os
from pathlib import Path

from sglang.srt.model_runner_plugins.api import MODEL_RUNNERS_GROUP

REFERENCE_RUNNER = "reference"
DISTRIBUTION = "sglang-reference-runner"


# The same runner, returning vocabulary-sharded logits for SGLang to gather.
SHARDED_REFERENCE_RUNNER = "reference-sharded"


def create_plugin():
    from sglang.test.model_runner_reference.plugin import ReferenceRunner

    return ReferenceRunner()


def create_sharded_plugin():
    from sglang.srt.model_runner_plugins.api import LogitsLayout
    from sglang.test.model_runner_reference.plugin import ReferenceRunner

    return ReferenceRunner(logits_layout=LogitsLayout.VOCAB_SHARDED)


def install_reference_runner(root: str) -> dict[str, str]:
    info = Path(root) / f"{DISTRIBUTION.replace('-', '_')}-1.0.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 1.0\n"
    )
    (info / "entry_points.txt").write_text(
        f"[{MODEL_RUNNERS_GROUP}]\n"
        f"{REFERENCE_RUNNER} = sglang.test.model_runner_reference:create_plugin\n"
        f"{SHARDED_REFERENCE_RUNNER} = "
        "sglang.test.model_runner_reference:create_sharded_plugin\n"
    )
    path = os.environ.get("PYTHONPATH")
    return {"PYTHONPATH": root if not path else f"{root}{os.pathsep}{path}"}


def read_step_trace(directory: str) -> list[dict]:
    """The records ``SGLANG_MODEL_RUNNER_STEP_TRACE_DIR`` collected, in step
    order, each with its saved tensors under ``"tensors"`` when there are any."""
    records = []
    for path in sorted(glob.glob(os.path.join(directory, "steps-*.jsonl"))):
        prefix = path[: -len(".jsonl")]
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            saved = f"{prefix}-{record['index']:05d}.pt"
            if os.path.exists(saved):
                import torch

                record["tensors"] = torch.load(saved)
            records.append(record)
    return records
