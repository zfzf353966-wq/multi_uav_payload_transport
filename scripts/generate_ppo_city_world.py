"""Keep the original logistics city and add the native PPO lidar follower."""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def generate_city_world(source: Path, output: Path):
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    if world is None:
        raise ValueError('Source SDF has no world')
    world.set('name', 'logistics_city_ppo')
    if not any(p.get('name') == 'mupt::SensorFollower' for p in world.findall('plugin')):
        ET.SubElement(world, 'plugin', filename='libmupt_sensor_follower.so',
                      name='mupt::SensorFollower')
    ET.indent(tree, space='  ')
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding='utf-8', xml_declaration=True)
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root / 'simulation/worlds/logistics_city.sdf')
    parser.add_argument('--output', type=Path, default=root / 'simulation/worlds/logistics_city_ppo.sdf')
    args = parser.parse_args()
    print(generate_city_world(args.source, args.output))


if __name__ == '__main__':
    main()
