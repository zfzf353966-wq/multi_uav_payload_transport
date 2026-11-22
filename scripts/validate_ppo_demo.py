#!/usr/bin/env python3
"""Run sequential, isolated Gazebo flights and audit completion and scan margin."""
import argparse
import csv
import json
import math
import subprocess
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path


def mount_metrics(run_dir):
    path = run_dir / 'gazebo_poses.jsonl'
    errors = []
    if path.exists():
        for line in path.read_text().splitlines():
            frame = json.loads(line)
            poses = {pose['name']: pose for pose in frame['poses']}
            if 'x500_1' in poses and 'uav_1_lidar' in poses:
                expected = list(poses['x500_1']['position'])
                expected[2] += .6
                errors.append(math.dist(expected, poses['uav_1_lidar']['position']))
    errors.sort()
    return dict(samples=len(errors),
                max_mount_error_m=max(errors) if errors else None,
                p95_mount_error_m=errors[int(.95 * (len(errors) - 1))] if errors else None)


def audit_run(run_dir, returncode, clearance):
    summary_path = run_dir / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    trace_path = run_dir / 'control_trace.csv'
    with trace_path.open() if trace_path.exists() else nullcontext([]) as stream:
        rows = list(csv.DictReader(stream))
    scans = [float(row['minimum_scan_m']) for row in rows]
    finite = [value for value in scans if math.isfinite(value)]
    low = [row for row in rows if float(row['minimum_scan_m']) < clearance]
    streak = longest = 0
    for scan in scans:
        streak = streak + 1 if scan < clearance else 0
        longest = max(longest, streak)
    spawned = (run_dir / 'spawn.json').exists()
    completed = returncode == 0 and summary.get('success') is True
    valid_scans = bool(scans) and len(finite) == len(scans)
    # A task PASS can still fail the stricter, single-sample scan-margin audit.
    margin_ok = valid_scans and min(finite) >= clearance
    return dict(run_dir=str(run_dir), returncode=returncode,
                obstacle_spawn_verified=spawned, task_completed=completed,
                scan_margin_passed=margin_ok,
                accepted=completed and spawned and margin_ok,
                clearance_threshold_m=clearance,
                minimum_cruise_scan_m=min(finite) if finite else None,
                low_scan_samples=len(low), longest_low_scan_streak=longest,
                low_scan_events=[{key: row[key] for key in
                    ('elapsed_s', 'phase', 'x_enu', 'y_enu', 'z_enu',
                     'minimum_scan_m', 'source', 'reason')} for row in low],
                abort_reason=summary.get('abort_reason'),
                lidar_mount=mount_metrics(run_dir),
                ppo_ticks=summary.get('all_ticks', {}).get('rl', 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--clearance', type=float, default=.8)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--environment', choices=('lab', 'city'), default='lab')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.scenarios) < 0 or not math.isfinite(args.clearance) or args.clearance <= 0:
        parser.error('repeats and clearance must be positive; scenarios non-negative')
    root = Path(__file__).resolve().parents[1]
    output = (args.output_dir or root / 'runs/direct_ppo_3d' /
              ('validation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(environment=args.environment, scenarios=args.scenarios, repeats=args.repeats, runs=[])
    for scenario in args.scenarios:
        for repeat in range(1, args.repeats + 1):
            run_dir = output / f'scenario_{scenario}_repeat_{repeat}'
            command = [str(root / 'scripts/run_ppo_demo.bash'), '--headless',
                       '--environment', args.environment,
                       '--scenario', str(scenario), '--output-dir', str(run_dir)]
            if args.model:
                command += ['--model', str(args.model.resolve())]
            print(f'Start scenario {scenario}, repeat {repeat}; {run_dir}', flush=True)
            with (output / f'scenario_{scenario}_repeat_{repeat}.log').open('w') as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
            audit = audit_run(run_dir, result.returncode, args.clearance)
            report['runs'].append(dict(scenario=scenario, repeat=repeat, **audit))
            report['task_completed_count'] = sum(r['task_completed'] for r in report['runs'])
            report['accepted_count'] = sum(r['accepted'] for r in report['runs'])
            (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
            print(f"Task={audit['task_completed']} scan_margin={audit['scan_margin_passed']} "
                  f"min_scan={audit['minimum_cruise_scan_m']}", flush=True)
    print(f"Accepted {report['accepted_count']}/{len(report['runs'])}; {output / 'validation.json'}")
    return 0 if all(r['accepted'] for r in report['runs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
