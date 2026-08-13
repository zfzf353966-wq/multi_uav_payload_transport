from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "scenario", "controller", "success", "duration_s", "path_length_m",
    "minimum_scan_m", "maximum_lateral_deviation_m", "rl_share_within_6m",
    "rl_share_within_4m", "fallback_ticks", "abort_reason",
)


def build_report(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("scenario_*/**/summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        scenario = summary_path.parts[-3].removeprefix("scenario_")
        rows.append(
            {
                "scenario": scenario,
                "controller": data["configured_source"],
                "success": data["success"],
                "duration_s": data["duration_s"],
                "path_length_m": data["path_length_m"],
                "minimum_scan_m": data["minimum_scan_m"],
                "maximum_lateral_deviation_m": data["maximum_lateral_deviation_m"],
                "rl_share_within_6m": data["rl_share_within_6m"],
                "rl_share_within_4m": data["rl_share_within_4m"],
                "fallback_ticks": data["all_ticks"]["fallback"],
                "abort_reason": data["abort_reason"],
            }
        )
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    aggregates = {}
    for controller in ("potential", "rl"):
        selected = [row for row in rows if row["controller"] == controller]
        successful = [row for row in selected if row["success"]]
        aggregates[controller] = {
            "runs": len(selected),
            "success_rate": len(successful) / len(selected) if selected else 0.0,
            "mean_duration_s": sum(float(row["duration_s"]) for row in successful) / len(successful) if successful else None,
            "mean_path_length_m": sum(float(row["path_length_m"]) for row in successful) / len(successful) if successful else None,
            "worst_minimum_scan_m": min((float(row["minimum_scan_m"]) for row in selected), default=None),
        }
    (root / "report.json").write_text(
        json.dumps({"runs": rows, "aggregate": aggregates}, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate avoidance A/B runs")
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    rows = build_report(arguments.root.resolve())
    print(f"Wrote {len(rows)} runs to {arguments.root}/comparison.csv and report.json")


if __name__ == "__main__":
    main()
