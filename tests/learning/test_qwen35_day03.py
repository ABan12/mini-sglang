"""Re-run real Engine verification in a fresh process; inspect execution and state isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def day3_report():
    project = Path(__file__).resolve().parents[2]
    model = os.environ.get("QWEN35_MODEL", "/root/rivermind-data/Qwen3.5-0.8B")
    artifacts = Path(os.environ.get("MINISGL_ARTIFACTS", "/root/rivermind-data/mini-sglang-results"))
    reference = os.environ.get("QWEN35_DAY3_REFERENCE", str(artifacts / "week02/day03/reference"))
    output = Path(os.environ.get("QWEN35_DAY3_TEST_OUTPUT", str(artifacts / "week02/day03/test-rerun")))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "verify-run.txt").open("w") as log:
        subprocess.run([sys.executable, str(project / "tests/learning/tools/qwen35_day03_verify.py"),
                        "--model", model, "--reference", reference, "--output", str(output)],
                       cwd=project, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output / "engine-comparison.json").read_text())


@pytest.mark.parametrize("mode", ["full", "step", "chunked"])
def test_all_position_logits_and_24_layer_states(day3_report, mode):
    selected = {key: value for key, value in day3_report["comparisons"].items()
                if key.startswith("A." + mode + ".")}
    assert len(selected) == 50  # All logits + actual last logits + 24 layers × two states.
    assert all(value["passed"] for value in selected.values())


def test_different_length_requests_survive_batch_reordering(day3_report):
    for request in ("A", "B"):
        selected = [value for key, value in day3_report["comparisons"].items()
                    if key.startswith(request + ".reordered.")]
        assert len(selected) == 50
        assert all(value["passed"] for value in selected)
    assert day3_report["checks"]["reordered_request_lengths"]


def test_reallocated_gdn_state_is_zero_and_does_not_change_other_request(day3_report):
    assert day3_report["checks"]["released_slot_reused"]
    # A fresh history reads as None; zeroing is verified directly on both physical GDN buffers.
    assert day3_report["checks"]["fresh_request_state_layout"]
    assert day3_report["checks"]["new_request_gdn_zero"]
    isolated = [value for key, value in day3_report["comparisons"].items()
                if key.startswith("B.unchanged_after_C.")]
    replayed = [value for key, value in day3_report["comparisons"].items()
                if key.startswith("C.reallocated.")]
    assert len(isolated) == 48 and all(value["exact"] for value in isolated)
    assert len(replayed) == 50 and all(value["passed"] for value in replayed)


def test_real_engine_sampler_and_eager_path(day3_report):
    assert day3_report["checks"]["sampler_matches_actual_logits"]
    assert day3_report["checks"]["cuda_graph_disabled"]
    assert day3_report["status"] == "passed"
