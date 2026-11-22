"""The city demo must retain the user's original city geometry and dynamics."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location('generate_city', ROOT / 'scripts/generate_ppo_city_world.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_city_generator_preserves_every_original_world_element(tmp_path):
    source = ROOT / 'simulation/worlds/logistics_city.sdf'
    original = source.read_bytes()
    output = tmp_path / 'logistics_city_ppo.sdf'
    MODULE.generate_city_world(source, output)
    city = ET.parse(source).getroot().find('world')
    generated = ET.parse(output).getroot().find('world')
    assert generated.get('name') == 'logistics_city_ppo'
    assert len(generated) == len(city) + 1
    for before, after in zip(city, generated):
        canonical = lambda node: ET.canonicalize(ET.tostring(node, encoding='unicode'), strip_text=True)
        assert canonical(before) == canonical(after)
    assert generated[-1].get('name') == 'mupt::SensorFollower'
    assert source.read_bytes() == original


def test_city_generator_does_not_duplicate_native_plugin(tmp_path):
    output = tmp_path / 'city.sdf'
    MODULE.generate_city_world(ROOT / 'simulation/worlds/logistics_city.sdf', output)
    MODULE.generate_city_world(output, output)
    world = ET.parse(output).getroot().find('world')
    assert sum(p.get('name') == 'mupt::SensorFollower' for p in world.findall('plugin')) == 1
