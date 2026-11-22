"""The batch audit must not equate task completion with adequate scan margin."""
import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / 'scripts/validate_ppo_demo.py'
SPEC = importlib.util.spec_from_file_location('validate_ppo_demo', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_run(root, scans):
    (root / 'summary.json').write_text(json.dumps({'success': True}))
    (root / 'spawn.json').write_text('{}')
    fields = ['elapsed_s', 'phase', 'x_enu', 'y_enu', 'z_enu',
              'minimum_scan_m', 'source', 'reason']
    with (root / 'control_trace.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scan in scans:
            writer.writerow(dict.fromkeys(fields, '') | {'minimum_scan_m': scan})


def test_single_low_sample_fails_margin_even_when_task_passes(tmp_path):
    make_run(tmp_path, [20, .634, 8])
    result = MODULE.audit_run(tmp_path, 0, .8)
    assert result['task_completed']
    assert not result['accepted']
    assert result['low_scan_samples'] == result['longest_low_scan_streak'] == 1


def test_verified_completion_and_margin_are_required(tmp_path):
    make_run(tmp_path, [2, 3, 4])
    assert MODULE.audit_run(tmp_path, 0, .8)['accepted']
    assert not MODULE.audit_run(tmp_path, 1, .8)['accepted']
    (tmp_path / 'spawn.json').unlink()
    assert not MODULE.audit_run(tmp_path, 0, .8)['accepted']


def test_missing_or_nonfinite_scans_cannot_pass(tmp_path):
    make_run(tmp_path, [])
    assert not MODULE.audit_run(tmp_path, 0, .8)['accepted']
    make_run(tmp_path, [float('nan')])
    assert not MODULE.audit_run(tmp_path, 0, .8)['accepted']


def test_mount_metrics_measure_actual_offset(tmp_path):
    assert MODULE.mount_metrics(tmp_path)['samples'] == 0
    frame = {'poses': [
        {'name': 'x500_1', 'position': [1, 2, 3]},
        {'name': 'uav_1_lidar', 'position': [1.3, 2, 3.6]},
    ]}
    (tmp_path / 'gazebo_poses.jsonl').write_text(json.dumps(frame) + '\n')
    assert abs(MODULE.mount_metrics(tmp_path)['max_mount_error_m'] - .3) < 1e-9
