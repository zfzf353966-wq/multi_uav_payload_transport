"""Derive a controlled PPO test world, retaining PX4 systems and lidar geometry."""
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
tree = ET.parse(root / "simulation/worlds/logistics_city.sdf")
world = tree.getroot().find("world")
world.set("name", "ppo_avoidance_lab")
keep = {"ground_plane", "roads", "uav_depot", "uav_1_lidar", "uav_2_lidar", "uav_3_lidar"}
for element in list(world):
    if element.tag == "include" or (element.tag == "model" and element.get("name") not in keep):
        world.remove(element)
ET.SubElement(world, 'plugin', filename='libmupt_sensor_follower.so',
              name='mupt::SensorFollower')
ET.indent(tree, space="  ")
output = root / "simulation/worlds/ppo_avoidance_lab.sdf"
tree.write(output, encoding="utf-8", xml_declaration=True)
print(output)
