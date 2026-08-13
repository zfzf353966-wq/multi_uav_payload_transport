from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path


OBSTACLES = (
    ("red", -6.0),
    ("orange", 0.0),
    ("purple", 6.0),
)


def scenario_parameters(scenario: int) -> list[dict[str, float | str]]:
    """Return deterministic dynamic-obstacle parameters for an integer seed."""
    generator = random.Random(47_311 + int(scenario))
    parameters = []
    for index, (name, nominal_x) in enumerate(OBSTACLES):
        centre_x = nominal_x + generator.uniform(-0.8, 0.8)
        centre_y = generator.uniform(-0.7, 0.7)
        radius = generator.uniform(1.8, 3.1)
        phase = generator.uniform(-math.pi, math.pi)
        speed = generator.uniform(0.28, 0.55)
        if (scenario + index) % 2:
            speed = -speed
        parameters.append(
            {
                "name": name,
                "centre_x": centre_x,
                "centre_y": centre_y,
                "radius": radius,
                "phase": phase,
                "speed": speed,
            }
        )
    return parameters


def generate_scenario(template: Path, output: Path, scenario: int) -> None:
    tree = ET.parse(template)
    root = tree.getroot()
    model = root.find("model")
    if model is None:
        raise ValueError("SDF template does not contain a model")
    model.set("name", f"complex_moving_obstacles_scenario_{scenario}")
    for item in scenario_parameters(scenario):
        name = str(item["name"])
        centre_x = float(item["centre_x"])
        centre_y = float(item["centre_y"])
        radius = float(item["radius"])
        phase = float(item["phase"])
        link = model.find(f"link[@name='{name}_sphere']/pose")
        joint = model.find(f"joint[@name='{name}_orbit']/pose")
        plugin = model.find(
            f"plugin[joint_name='{name}_orbit']/initial_velocity"
        )
        if link is None or joint is None or plugin is None:
            raise ValueError(f"SDF template is missing elements for {name}")
        link_x = centre_x + radius * math.cos(phase)
        link_y = centre_y + radius * math.sin(phase)
        link.text = f"{link_x:.6f} {link_y:.6f} 0 0 0 0"
        joint.text = f"{centre_x:.6f} {centre_y:.6f} 0 0 0 0"
        plugin.text = f"{float(item['speed']):.6f}"
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible obstacle SDF")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", type=int, required=True)
    arguments = parser.parse_args()
    generate_scenario(arguments.template, arguments.output, arguments.scenario)


if __name__ == "__main__":
    main()
