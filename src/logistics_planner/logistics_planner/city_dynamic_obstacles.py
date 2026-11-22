"""Seeded whole-city 3D obstacle orbits with conservative swept-volume checks."""
import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import yaml


def orbit_position(spec, seconds):
    c, a, w, p = (np.asarray(spec[key], dtype=float) for key in ('center', 'amplitude', 'rate', 'phase'))
    return c + a * np.array([math.cos(w[0] * seconds + p[0]),
                             math.sin(w[1] * seconds + p[1]), math.sin(w[2] * seconds + p[2])])


def static_boxes(config):
    boxes = []
    for o in config['static_obstacles']:
        c, h = np.array(o['center']), np.array(o['size']) / 2
        boxes.append((c-h, c+h))
    for x, y, z in config['tree_positions']:
        boxes.append((np.array([x-1.55, y-1.55, z]), np.array([x+1.55, y+1.55, z+5.55])))
    return boxes


def orbit_is_safe(spec, config, margin=.5):
    centre, amplitude = np.array(spec['center']), np.array(spec['amplitude'])
    reach = amplitude + spec['radius'] + margin
    low, high = centre - reach, centre + reach
    bounds = config['world']['bounds']
    if np.any(low < np.array(bounds['min'])) or np.any(high > np.array(bounds['max'])):
        return False
    return all(np.any(high < box_low) or np.any(low > box_high)
               for box_low, box_high in static_boxes(config))


def generate_specs(config, plan, *, seed=1, random_count=8, departure_count=2):
    """按种子生成出发通路及全城三维运动球；检查完整轨道的静态净空。"""
    rng = np.random.default_rng(seed)
    specs = []
    # Sample anchors along initial A* legs, not a fixed south-corridor location.
    anchors = []
    for route in plan['routes']:
        if not route.get('delivery_legs'):
            continue
        path = route['delivery_legs'][0]['path']
        for first, last in zip(path, path[1:]):
            d = np.linalg.norm(np.array(last)-first)
            if d < 2:
                continue
            for fraction in np.linspace(.2, .7, 6):
                point = np.array(first) + fraction * (np.array(last)-first)
                if np.linalg.norm(point-np.array(path[0])) <= 12 and min(np.linalg.norm(point-np.array(s)) for s in config['depot']['uav_spawn_points']) > 5:
                    anchors.append(point)
    rng.shuffle(anchors)
    for index in range(departure_count + random_count):
        departure = index < departure_count
        accepted = None
        for attempt in range(3000):
            if departure:
                if not anchors:
                    raise ValueError('No departure A* anchors available')
                centre = anchors[attempt % len(anchors)]
            else:
                quadrant = (index - departure_count) % 4
                centre = np.array([rng.uniform(-34, -3) if quadrant % 2 == 0 else rng.uniform(3, 34),
                                   rng.uniform(-24, -3) if quadrant < 2 else rng.uniform(3, 24),
                                   rng.uniform(7, 30)])
            horizontal = float(rng.uniform(.6, 1.3) if departure else rng.uniform(.8, 2.0))
            omega = float(rng.uniform(.18, .4)) * rng.choice([-1., 1.])
            spec = dict(name=f'city_dynamic_{index+1}', category='departure' if departure else 'city_random',
                        center=centre.tolist(), radius=float(rng.uniform(.45, .7)),
                        amplitude=[horizontal, horizontal, float(rng.uniform(.35, 1.0))],
                        rate=[omega, omega, float(rng.uniform(.2, .5))],
                        phase=[float(rng.uniform(-math.pi, math.pi)), 0., float(rng.uniform(-math.pi, math.pi))])
            spec['phase'][1] = spec['phase'][0]
            if not orbit_is_safe(spec, config):
                continue
            # Preserve clearance from the original city's existing orbit too.
            legacy_reach=np.array([4.,4.,0.])+.45+.3
            if np.all(np.abs(centre-np.array([10.,0.,8.])) < legacy_reach+np.array(spec['amplitude'])+spec['radius']+.3):
                continue
            reach = np.array(spec['amplitude']) + spec['radius'] + .6
            if any(np.all(centre[:2]-reach[:2] < np.array(spawn[:2])+.6)
                   and np.all(centre[:2]+reach[:2] > np.array(spawn[:2])-.6)
                   and centre[2]-reach[2] < 4.0+7.5*i
                   for i, spawn in enumerate(config['depot']['uav_spawn_points'])):
                continue
            if any(np.all(np.abs(np.array(old['center'])-centre) <
                          np.array(old['amplitude'])+np.array(spec['amplitude'])+old['radius']+spec['radius']+.3)
                   for old in specs):
                continue
            accepted = spec
            break
        if accepted is None:
            raise ValueError(f'Cannot place safe obstacle {index+1}')
        specs.append(accepted)
    return specs


def write_world(source, output, specs):
    tree = ET.parse(source)
    world = tree.getroot().find('world')
    world.set('name', 'logistics_delivery_ppo')
    for include in list(world.findall('include')):
        if include.findtext('uri')!='model://delivery_package':continue
        package=ET.parse(Path(source).parents[1]/'models/delivery_package/model.sdf').getroot().find('model')
        package.set('name',include.findtext('name'))
        ET.SubElement(package,'pose').text=include.findtext('pose')
        for visual in package.findall('.//visual'):ET.SubElement(visual,'visibility_flags').text='2'
        index=list(world).index(include);world.remove(include);world.insert(index,package)
    for ray in world.findall('.//sensor/ray'):
        ET.SubElement(ray,'visibility_mask').text=str(0xFFFFFFFD)
    # Keep the trained three-layer observation unchanged. The independent
    # safety cloud covers steep ascent/descent and is not a policy input.
    for i in range(1,4):
        link=world.find(f"model[@name='uav_{i}_lidar']/link")
        safety=copy.deepcopy(link.find("sensor[@name='scan']"))
        safety.set('name','safety_scan')
        safety.find('topic').text=f'/mupt/uav_{i}/safety_scan'
        vertical=safety.find('ray/scan/vertical')
        vertical.find('samples').text='11'
        vertical.find('min_angle').text=str(-math.pi/3)
        vertical.find('max_angle').text=str(math.pi/3)
        safety.find('ray/range/min').text='.9'
        safety.find('ray/range/max').text='8'
        link.append(safety)
    for name, filename in [('mupt::SensorFollower', 'libmupt_sensor_follower.so'),
                           ('mupt::CityObstacles', 'libmupt_city_obstacles.so'),
                           ('mupt::PackageFollower','libmupt_package_follower.so')]:
        if not any(p.get('name') == name for p in world.findall('plugin')):
            ET.SubElement(world, 'plugin', name=name, filename=filename)
    plugin = next(p for p in world.findall('plugin') if p.get('name') == 'mupt::CityObstacles')
    for spec in specs:
        model = ET.SubElement(world, 'model', name=spec['name'])
        ET.SubElement(model, 'static').text = 'true'
        ET.SubElement(model, 'pose').text = ' '.join(map(str, orbit_position(spec, 0))) + ' 0 0 0'
        link = ET.SubElement(model, 'link', name='sphere')
        for kind in ('collision', 'visual'):
            element = ET.SubElement(link, kind, name=kind)
            ET.SubElement(ET.SubElement(ET.SubElement(element, 'geometry'), 'sphere'), 'radius').text = str(spec['radius'])
            if kind == 'visual':
                material = ET.SubElement(element, 'material')
                for field in ('ambient', 'diffuse'):
                    ET.SubElement(material, field).text = '0.95 0.2 0.08 1' if spec['category']=='departure' else '0.8 0.12 0.8 1'
        obstacle = ET.SubElement(plugin, 'obstacle')
        ET.SubElement(obstacle, 'name').text = spec['name']
        for key in ('center', 'amplitude', 'rate', 'phase'):
            ET.SubElement(obstacle, key).text = ' '.join(map(str, spec[key]))
    ET.indent(tree, space='  ')
    tree.write(output, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--random-count', type=int, default=8)
    parser.add_argument('--departure-count', type=int, default=2)
    args = parser.parse_args()
    if args.random_count < 4 or args.departure_count not in (1, 2):
        parser.error('use at least four city obstacles and one or two departure obstacles')
    config = yaml.safe_load(args.config.read_text())
    specs = generate_specs(config, json.loads(args.plan.read_text()), seed=args.seed,
                           random_count=args.random_count, departure_count=args.departure_count)
    write_world(args.source, args.output, specs)
    args.manifest.write_text(json.dumps(dict(seed=args.seed, static_margin_m=.5, obstacles=specs), indent=2)+'\n')


if __name__ == '__main__':
    main()
