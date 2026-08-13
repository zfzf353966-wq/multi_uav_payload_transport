import json

from logistics_planner.avoidance_batch_report import build_report
from logistics_planner.obstacle_scenario import scenario_parameters


def test_scenario_parameters_are_reproducible_and_vary() -> None:
    assert scenario_parameters(3) == scenario_parameters(3)
    assert scenario_parameters(3) != scenario_parameters(4)
    assert {item["speed"] > 0 for item in scenario_parameters(3)} == {True, False}


def test_batch_report_aggregates_two_controllers(tmp_path) -> None:
    for controller, duration in (("potential", 100.0), ("rl", 75.0)):
        output = tmp_path / "scenario_01" / controller
        output.mkdir(parents=True)
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "configured_source": controller,
                    "success": True,
                    "duration_s": duration,
                    "path_length_m": 60.0,
                    "minimum_scan_m": 1.5,
                    "maximum_lateral_deviation_m": 2.0,
                    "rl_share_within_6m": 0.8 if controller == "rl" else 0.0,
                    "rl_share_within_4m": 0.7 if controller == "rl" else 0.0,
                    "all_ticks": {"fallback": 3 if controller == "rl" else 0},
                    "abort_reason": None,
                }
            ),
            encoding="utf-8",
        )
    rows = build_report(tmp_path)
    assert len(rows) == 2
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["aggregate"]["rl"]["mean_duration_s"] == 75.0
