"""Day 2: run the actual loader/layers against independent same-input references."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from minisgl.layers.deltanet import recurrent_delta_rule


def test_delta_rule_updates_the_matching_key_row():
    # Two orthogonal keys: writing key 1 must preserve the key 2 association.
    q = torch.tensor([[[[1.,0.]],[[0.,1.]],[[0.,1.]]]])
    k = torch.tensor([[[[1.,0.]],[[0.,1.]],[[1.,0.]]]])
    v = torch.tensor([[[[2.,4.]],[[3.,1.]],[[0.,2.]]]])
    g = torch.tensor([[[0.],[-0.6931471805599453],[0.]]])
    beta = torch.ones(1,3,1)
    output,state = recurrent_delta_rule(q,k,v,g,beta)
    expected_state = torch.tensor([[[[0.,2.],[3.,1.]]]])
    expected_output = torch.tensor([[[[2.,4.]],[[3.,1.]],[[3.,1.]]]]) / (2.**0.5)
    torch.testing.assert_close(state,expected_state,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(output,expected_output,atol=1e-5,rtol=1e-5)


@pytest.fixture(scope='module')
def comparison():
    project = Path(__file__).resolve().parents[2]
    model = os.environ.get('QWEN35_MODEL','/root/rivermind-data/Qwen3.5-0.8B')
    reference = os.environ.get('QWEN35_DAY2_REFERENCE',str(project/'results/week02/day02/reference'))
    output = project/'results/week02/day02'
    with (output/'verify-run.txt').open('w') as log:
        subprocess.run([sys.executable,str(project/'experiments/qwen35_day02_verify.py'),
                        '--model',model,'--reference',reference,'--output',str(output)],
                       cwd=project,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((output/'layer-comparison.json').read_text()),json.loads((output/'weight-load-coverage.json').read_text())


def test_complete_text_weight_coverage(comparison):
    _,coverage = comparison
    assert coverage['status'] == 'passed'
    assert coverage['checkpoint_text_keys'] == 320
    assert coverage['runtime_text_keys'] == 296
    assert len(coverage['original_fp32_keys']) == 36
    assert coverage['vision_skipped'] == 153
    assert coverage['mtp_skipped'] == 15


@pytest.mark.parametrize('dtype',['float32','bfloat16'])
@pytest.mark.parametrize('layer',['layer_00','layer_03'])
def test_outputs_norm_mlp_rope_and_final_states(comparison,dtype,layer):
    report,_ = comparison
    selected = {k:v for k,v in report['comparisons'].items() if k.startswith(dtype+'.'+layer+'.')}
    assert selected
    assert all(v['passed'] for v in selected.values()),[k for k,v in selected.items() if not v['passed']]
